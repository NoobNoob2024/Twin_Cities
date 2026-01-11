export type AudioCaptureResult = {
  samples: Float32Array;
  sampleRate: number;
};

export async function recordMono(durationMs: number): Promise<AudioCaptureResult> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  // ScriptProcessor is deprecated but widely supported and fine for a demo.
  const processor = audioContext.createScriptProcessor(4096, 1, 1);

  const chunks: Float32Array[] = [];
  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));
  };

  source.connect(processor);
  // connect to destination to keep processing alive in some browsers
  processor.connect(audioContext.destination);

  await new Promise<void>((resolve) => setTimeout(resolve, durationMs));

  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((t) => t.stop());
  await audioContext.close();

  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const samples = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) {
    samples.set(c, offset);
    offset += c.length;
  }

  return { samples, sampleRate: audioContext.sampleRate };
}

export async function decodeAudioFileToMono(file: File): Promise<AudioCaptureResult> {
  const buf = await file.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(buf.slice(0));
    const channels = decoded.numberOfChannels;
    const len = decoded.length;

    if (channels === 1) {
      return { samples: decoded.getChannelData(0).slice(), sampleRate: decoded.sampleRate };
    }

    const out = new Float32Array(len);
    for (let c = 0; c < channels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < len; i++) out[i] += ch[i] ?? 0;
    }
    for (let i = 0; i < len; i++) out[i] /= channels;
    return { samples: out, sampleRate: decoded.sampleRate };
  } finally {
    await audioContext.close();
  }
}

export function resampleLinear(input: Float32Array, inputRate: number, outputRate: number): Float32Array {
  if (inputRate === outputRate) return input;
  const ratio = outputRate / inputRate;
  const outputLength = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(outputLength);
  for (let i = 0; i < outputLength; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] ?? 0;
    const b = input[idx + 1] ?? a;
    output[i] = a + (b - a) * frac;
  }
  return output;
}

export function takeOrPadToLength(input: Float32Array, targetLength: number): Float32Array {
  if (input.length === targetLength) return input;
  const out = new Float32Array(targetLength);
  if (input.length >= targetLength) {
    out.set(input.subarray(0, targetLength));
  } else {
    out.set(input);
  }
  return out;
}

export function takeCenterSegment(input: Float32Array, targetLength: number): Float32Array {
  if (input.length <= targetLength) return takeOrPadToLength(input, targetLength);
  const start = Math.floor((input.length - targetLength) / 2);
  return input.subarray(start, start + targetLength);
}
