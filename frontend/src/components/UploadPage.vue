<template>
  <div>{{ status }}</div>
</template>

<script lang="ts">
export default {
  props: {
    images: {
      type: Array,
      required: true
    },
    mask_color: {
      type: String,
      required: false
    },
    is_mosaic: {
      type: Boolean,
      required: true
    },
    variations: {
      type: Number,
      required: true
    },
    output: {
      type: String,
      required: true
    }
  },
  created() {
    this.process();
  },
  data() {
    return {
      status: "upload",
      info: null,
      queuePos: null
    };
  },
  methods: {
    setState(status, info = null) {
      if (status === "queue") {
        this.queuePos = info;
      } else {
        this.info = info
      }
      this.status = status;
    },
    process() {
      let buffer = new Uint8Array();
      const processChunk = (value: Uint8Array<ArrayBufferLike>) => {
        const newBuffer = new Uint8Array(buffer.length + value.length);
        newBuffer.set(buffer);
        newBuffer.set(value, buffer.length);
        buffer = newBuffer;

        while (buffer.length >= 5) {
          const dataSize = new DataView(buffer.buffer).getUint32(1, false);
          const totalSize = 5 + dataSize;
          if (buffer.length < totalSize) {
            break;
          }

          const statusCode = buffer[0];
          const decoder = new TextDecoder('utf-8');
          const data = buffer.slice(5, totalSize);
          switch (statusCode) {
            case 200:
              this.setState("done");
              break;
            case 201:
              this.setState("cancel")
              break;
            case 202:
              this.setState("error-internal")
              break;
            case 1:
              this.setState("start-image")
              console.log(decoder.decode(data)); // 2 ints pos, end
              break;
            case 2:
              this.setState("end-image");
              console.log(decoder.decode(data)); // 2 ints pos, end
              break;
            case 3:
              this.setState("apply-variant");
              console.log(decoder.decode(data)); // 1 int
              break;
            case 4:
              this.setState("generate-mask");
              break;
            case 5:
              this.setState("finished");
              break;
            case 6:
              this.setState("remove-alpha");
              break;
            case 7:
              this.setState("find-regions");
              break;
            case 8:
              this.setState("decensor-segment");
              console.log(decoder.decode(data)); // 2 ints pos, end
              break;
            case 9:
              this.setState("restore-alpha");
              break;
            case 10:
              this.setState("unknown-info");
              console.log(decoder.decode(data)); // message
              break;
            case 11:
              this.setState("found-regions");
              console.log(decoder.decode(data)); // 1 int
              break;
            case 12:
              this.setState("unknown-debug");
              console.log(decoder.decode(data)); // message
              break;
            case 13:
              this.setState("no-regions");
              break;
            case 14:
              this.setState("missing-model");
              break;
            case 15:
              this.setState("bounding-box-out-of-bounds");
              break;
            case 16:
              this.setState("error-unknown");
              console.log(decoder.decode(data)); // message
              break;
            case 17:
              this.setState("warn-unknown");
              console.log(decoder.decode(data)); // message
              break;
            case 198:
              this.setState('queue', decoder.decode(data));
              break;
            case 199:
              this.setState('queue');
              this.queuePos = null;
              break;
          }
          buffer = buffer.slice(totalSize);
        }
      };

      function hexToRgb(hex: string) {
        hex = hex.replace(/^#/, '');
        if (hex.length === 3) {
          hex = hex.split('').map(c => c + c).join('');
        }
        const num = parseInt(hex, 16);
        return `${(num >> 16) & 255},${(num >> 8) & 255},${num & 255}`;
      }

      const uploadWithProgress = async () => {
        try {

          const response = await fetch(`http://127.0.0.1:8000/decensor`, {
            method: 'POST',
            headers: {
              "Content-Type": "application/json",
            },


            body: JSON.stringify({
              imgs: this.images.map(v => {
                return {
                  mask: this.mask_color ? "rgb-" + hexToRgb(this.mask_color) : "file-" + v.mask,
                  output_name: v.image.name,
                  img_id: v.image.id,
                  variations: this.variations,
                  is_mosaic: this.is_mosaic,
                  output: this.output
                }
              })
            }),
          });

          if (response.status !== 200 || !response.body) {
            this.setState('upload');
            return;
          }

          const reader = response.body.getReader();
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            processChunk(value);
          }
        } catch (error) {
          console.error(error);
          this.setState('error-disconnect');
        }
      }
      uploadWithProgress()
    }
  }
};
</script>
