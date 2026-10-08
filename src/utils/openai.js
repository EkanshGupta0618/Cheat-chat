const { getSystemPrompt } = require('./prompts');
const { sendToRenderer, initializeNewSession, saveConversationTurn } = require('./gemini');
const { getCredentials } = require('../storage');

const OPENAI_BASE_URL = 'https://api.openai.com/v1';
const SAMPLE_RATE = 24000;

let apiKey = null;
let chatModel = null;
let transcriptionModel = null;
let conversationHistory = [];
let currentSystemPrompt = null;
let isOpenAiActive = false;

let isSpeaking = false;
let speechBuffers = [];
let silenceFrameCount = 0;
let speechFrameCount = 0;

// Same energy-based VAD settings as local mode (VERY_AGGRESSIVE)
const vadConfig = { energyThreshold: 0.02, speechFramesRequired: 2, silenceFramesRequired: 15 };

function calculateRms(pcm16Buffer) {
    const samples = pcm16Buffer.length / 2;
    if (samples === 0) return 0;

    let sumSquares = 0;
    for (let i = 0; i < samples; i++) {
        const sample = pcm16Buffer.readInt16LE(i * 2) / 32768;
        sumSquares += sample * sample;
    }

    return Math.sqrt(sumSquares / samples);
}

function processVad(pcm16Buffer) {
    const rms = calculateRms(pcm16Buffer);
    const isVoice = rms > vadConfig.energyThreshold;

    if (isVoice) {
        speechFrameCount += 1;
        silenceFrameCount = 0;

        if (!isSpeaking && speechFrameCount >= vadConfig.speechFramesRequired) {
            isSpeaking = true;
            speechBuffers = [];
            console.log('[OpenAI] Speech started (RMS:', rms.toFixed(4), ')');
            sendToRenderer('update-status', 'Listening... (speech detected)');
        }
    } else {
        silenceFrameCount += 1;
        speechFrameCount = 0;

        if (isSpeaking && silenceFrameCount >= vadConfig.silenceFramesRequired) {
            isSpeaking = false;
            const audioData = Buffer.concat(speechBuffers);
            speechBuffers = [];
            console.log('[OpenAI] Speech ended, accumulated', audioData.length, 'bytes');
            sendToRenderer('update-status', 'Transcribing...');
            handleSpeechEnd(audioData);
            return;
        }
    }

    if (isSpeaking) {
        speechBuffers.push(Buffer.from(pcm16Buffer));
    }
}

function createWavBuffer(pcm16Buffer) {
    const header = Buffer.alloc(44);
    const byteRate = SAMPLE_RATE * 2;

    header.write('RIFF', 0);
    header.writeUInt32LE(36 + pcm16Buffer.length, 4);
    header.write('WAVE', 8);
    header.write('fmt ', 12);
    header.writeUInt32LE(16, 16);
    header.writeUInt16LE(1, 20);
    header.writeUInt16LE(1, 22);
    header.writeUInt32LE(SAMPLE_RATE, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(2, 32);
    header.writeUInt16LE(16, 34);
    header.write('data', 36);
    header.writeUInt32LE(pcm16Buffer.length, 40);

    return Buffer.concat([header, pcm16Buffer]);
}

async function readErrorMessage(response) {
    const errorText = await response.text();
    try {
        return JSON.parse(errorText).error?.message || errorText;
    } catch {
        return errorText;
    }
}

async function transcribeAudio(pcm16Buffer) {
    const formData = new FormData();
    formData.append('file', new Blob([createWavBuffer(pcm16Buffer)], { type: 'audio/wav' }), 'speech.wav');
    formData.append('model', transcriptionModel);
    formData.append('response_format', 'json');

    const response = await fetch(`${OPENAI_BASE_URL}/audio/transcriptions`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${apiKey}` },
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`OpenAI transcription failed (HTTP ${response.status}): ${await readErrorMessage(response)}`);
    }

    const result = await response.json();
    const text = result.text?.trim() || '';
    console.log('[OpenAI] Transcription:', text);
    return text;
}

async function handleSpeechEnd(audioData) {
    if (!isOpenAiActive) return;

    // Skip clips shorter than ~0.5s
    if (audioData.length < SAMPLE_RATE) {
        console.log('[OpenAI] Audio too short, skipping');
        sendToRenderer('update-status', 'Listening...');
        return;
    }

    try {
        const transcription = await transcribeAudio(audioData);

        if (!transcription || transcription.length < 2) {
            console.log('[OpenAI] Empty transcription, skipping');
            sendToRenderer('update-status', 'Listening...');
            return;
        }

        sendToRenderer('update-status', 'Generating response...');
        await sendToChat(transcription);
    } catch (error) {
        console.error('[OpenAI] Error:', error);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
    }
}

async function readStreamingResponse(response, onText) {
    const decoder = new TextDecoder();
    let pendingText = '';
    let fullText = '';

    for await (const chunk of response.body) {
        pendingText += decoder.decode(chunk, { stream: true });
        const lines = pendingText.split('\n');
        pendingText = lines.pop() || '';

        for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            const data = line.slice(6).trim();
            if (!data || data === '[DONE]') continue;

            const event = JSON.parse(data);
            const token = event.choices?.[0]?.delta?.content || '';
            if (!token) continue;

            fullText += token;
            onText(fullText);
        }
    }

    return fullText;
}

async function requestChat(messages, onText) {
    const response = await fetch(`${OPENAI_BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
            model: chatModel,
            messages,
            stream: true,
            max_completion_tokens: 2048,
        }),
    });

    if (!response.ok || !response.body) {
        throw new Error(`OpenAI returned HTTP ${response.status}: ${await readErrorMessage(response)}`);
    }

    return readStreamingResponse(response, onText);
}

function pushHistory(message) {
    conversationHistory.push(message);
    if (conversationHistory.length > 20) {
        conversationHistory = conversationHistory.slice(-20);
    }
}

async function streamReply(messages, userText) {
    let isFirst = true;
    const fullText = await requestChat(messages, text => {
        sendToRenderer(isFirst ? 'new-response' : 'update-response', text);
        isFirst = false;
    });

    if (fullText.trim()) {
        pushHistory({ role: 'assistant', content: fullText.trim() });
        saveConversationTurn(userText, fullText);
    }

    sendToRenderer('update-status', 'Listening...');
    return fullText;
}

async function sendToChat(text) {
    pushHistory({ role: 'user', content: text.trim() });
    const messages = [{ role: 'system', content: currentSystemPrompt }, ...conversationHistory];
    return streamReply(messages, text);
}

async function initializeOpenAiSession(model, sttModel, profile, customPrompt) {
    console.log('[OpenAI] Initializing session:', { model, sttModel, profile });
    sendToRenderer('session-initializing', true);

    try {
        closeOpenAiSession();

        apiKey = (getCredentials().openaiKey || '').trim();
        if (!apiKey) {
            throw new Error('No OpenAI API key configured');
        }

        // Verify the key and model before starting capture
        const response = await fetch(`${OPENAI_BASE_URL}/models/${encodeURIComponent(model)}`, {
            headers: { Authorization: `Bearer ${apiKey}` },
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await readErrorMessage(response)}`);
        }

        chatModel = model;
        transcriptionModel = sttModel;
        currentSystemPrompt = getSystemPrompt(profile, customPrompt, false);
        conversationHistory = [];
        isSpeaking = false;
        speechBuffers = [];
        silenceFrameCount = 0;
        speechFrameCount = 0;

        initializeNewSession(profile, customPrompt);
        isOpenAiActive = true;
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'OpenAI ready - Listening...');
        return true;
    } catch (error) {
        console.error('[OpenAI] Initialization error:', error);
        closeOpenAiSession();
        sendToRenderer('session-initializing', false);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
        return false;
    }
}

function processOpenAiAudio(monoChunk24k) {
    if (!isOpenAiActive) return;
    processVad(monoChunk24k);
}

function closeOpenAiSession() {
    isOpenAiActive = false;
    apiKey = null;
    chatModel = null;
    transcriptionModel = null;
    isSpeaking = false;
    speechBuffers = [];
    silenceFrameCount = 0;
    speechFrameCount = 0;
    conversationHistory = [];
    currentSystemPrompt = null;
}

async function sendOpenAiText(text) {
    if (!isOpenAiActive) {
        return { success: false, error: 'No active OpenAI session' };
    }

    try {
        sendToRenderer('update-status', 'Generating response...');
        await sendToChat(text);
        return { success: true };
    } catch (error) {
        console.error('[OpenAI] Text error:', error);
        sendToRenderer('update-status', 'OpenAI error: ' + error.message);
        return { success: false, error: error.message };
    }
}

async function sendOpenAiImage(base64Data, prompt) {
    if (!isOpenAiActive) {
        return { success: false, error: 'No active OpenAI session' };
    }

    const userMessage = {
        role: 'user',
        content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64Data}` } },
        ],
    };

    try {
        sendToRenderer('update-status', 'Analyzing image...');
        const messages = [{ role: 'system', content: currentSystemPrompt }, ...conversationHistory, userMessage];
        // Only the text prompt is kept in history to avoid resending screenshots
        pushHistory({ role: 'user', content: prompt });
        const fullText = await streamReply(messages, prompt);
        return { success: true, text: fullText, model: chatModel };
    } catch (error) {
        console.error('[OpenAI] Image error:', error);
        sendToRenderer('update-status', 'OpenAI image error: ' + error.message);
        return { success: false, error: error.message };
    }
}

module.exports = {
    initializeOpenAiSession,
    processOpenAiAudio,
    closeOpenAiSession,
    sendOpenAiText,
    sendOpenAiImage,
};
