class TalkCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const configuredChunkSamples = options.processorOptions?.chunkSamples;
        this.chunkSamples = Number.isInteger(configuredChunkSamples) && configuredChunkSamples > 0
            ? configuredChunkSamples
            : 8192;
        this.sampleBuffer = new Float32Array(this.chunkSamples);
        this.writeIndex = 0;
    }

    process(inputs, outputs) {
        const input = inputs[0];
        const inputChannel = input && input[0];

        if (inputChannel && inputChannel.length > 0) {
            let offset = 0;
            while (offset < inputChannel.length) {
                const copyCount = Math.min(
                    inputChannel.length - offset,
                    this.chunkSamples - this.writeIndex,
                );
                this.sampleBuffer.set(inputChannel.subarray(offset, offset + copyCount), this.writeIndex);
                this.writeIndex += copyCount;
                offset += copyCount;

                if (this.writeIndex === this.chunkSamples) {
                    let sumSquares = 0;
                    const pcm16 = new Int16Array(this.chunkSamples);
                    for (let index = 0; index < this.chunkSamples; index += 1) {
                        const sample = Math.max(-1, Math.min(1, this.sampleBuffer[index]));
                        sumSquares += sample * sample;
                        pcm16[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
                    }

                    this.port.postMessage(
                        {
                            type: 'chunk',
                            pcm16: pcm16.buffer,
                            rms: Math.sqrt(sumSquares / this.chunkSamples),
                        },
                        [pcm16.buffer],
                    );
                    this.writeIndex = 0;
                }
            }
        }

        const output = outputs[0];
        if (output) {
            for (const channel of output) {
                channel.fill(0);
            }
        }

        return true;
    }
}

registerProcessor('talk-capture-processor', TalkCaptureProcessor);