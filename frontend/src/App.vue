<script>
import DropOver from "@/components/DropOver.vue";
import Button from '@/components/Button.vue'
import ViewFiles from "@/components/ViewFiles.vue";
import SubmitButton from "@/components/SubmitButton.vue";
import UploadPage from "@/components/UploadPage.vue";

export default {
  components: {
    UploadPage,
    SubmitButton,
    ViewFiles,
    DropOver,
    Button
  },
  data() {
    return {
      files: [],
      mode: 0,
      is_mosaic: false,
      variations: 1,
      output: "decensor_output",
      color: '#00FF00',
      mask_suffix: "_mask",
      waiting: false
    };
  },
  methods: {
    deleteImageServer(id) {
      if (id) {
        fetch("http://127.0.0.1:8000/image", {method: "DELETE"})
      }
    },
    removeFile(file) {
      if (file.image) {
        this.deleteImageServer(file.image.id);
        this.files = this.files.filter(f => f !== file.image);
      }
      if (file.mask) {
        this.deleteImageServer(file.mask.id);
        this.files = this.files.filter(f => f !== file.mask);
      }
    },
    getFilesJoinedM() {
      if (this.mode === 1) {
        const fileMap = new Map();

        this.files.forEach(file => {
          let match, baseName, isMask;
          if (this.mask_suffix.length > 0) {
            const regex = new RegExp(`^(.*?)(_${this.mask_suffix})?(\\.[^.]+)$`);
            match = file.image.name.match(regex);
            if (match) {
              baseName = match[1];
              isMask = Boolean(match[2]);
            }
          } else {
            match = file.image.name.match(/^(.*?)(\.[^.]+)$/);
            if (match) {
              baseName = match[1];
              isMask = false;
            }
          }
          if (match) {
            if (!fileMap.has(baseName)) {
              fileMap.set(baseName, {image: null, mask: null, warn: true});
            }

            if (isMask) {
              fileMap.get(baseName).mask = file;
              fileMap.get(baseName).warn = false;
            } else {
              fileMap.get(baseName).image = file;
            }
          }
        });

        return Array.from(fileMap.values());
      } else {
        return this.files.map(v => {
          return {image: v, mask: null, warn: false}
        })
      }
    },
    handleFiles(files_) {
      const formData = new FormData();
      files_.forEach(file => formData.append("files", file));
      let files = files_.map(item => {
        item.id = null;
        return item;
      });
      this.files.push(...files);
      fetch("http://127.0.0.1:8000/images", {
        method: "POST",
        body: formData
      }).then(r => r.json()
      ).then(ids => {
        if (Array.isArray(ids) && ids.length === files.length) {
          files.forEach((file, index) => {
            file.id = ids[index];
          });
          this.files = [...this.files]
        } else {
          this.files = []
        }
      })
        .catch(() => {
          this.files = []
        });
    }
  },
  computed: {
    get_mask_color() {
      return this.mode === 0 ? this.color : null;
    },
    submitDisabled() {
      return this.getFilesJoinedM().some(v => v.warn || v.image.id === null);
    },
    getFilesJoined() {
      return this.getFilesJoinedM();
    }
  }
};
</script>

<template>
  <div class="bg-gray-100 h-screen w-screen">
    <div v-if="waiting" class="w-full, h-full">
      <UploadPage :images="getFilesJoined" :is_mosaic="is_mosaic" :mask_color="get_mask_color"
                  :output="output" :variations="variations"/>
    </div>
    <div v-else class="w-full, h-full">
      <DropOver v-if="files.length === 0" @files-selected="handleFiles"/>
      <div v-if="files.length !== 0" class="flex flex-col p-4">
        <header class="w-full flex justify-between mb-2">
          <div>
            <Button :active="mode===0" :left="true" content="RGB" @update:active="() => mode = 0"/>
            <Button :active="mode===1" content="Mask" @update:active="() => mode = 1"/>
            <Button :active="mode===2" :disabled="true" :right="true" content="Detect"
                    @update:active="() => mode = 2"/>
          </div>
          <div class="relative w-fit group flex">
            <select id="variations" v-model="variations" class="w-12 block p-2 border border-gray-300 bg-white rounded shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    name="variations">
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>

            <div
              class="absolute bottom-full left-1/2 transform -translate-x-[90%]  translate-y-[90%] mb-2 hidden group-hover:block w-max bg-gray-900 text-white text-xs rounded py-1 px-2 shadow-lg">
              The amount of variations to generate for each image
            </div>
          </div>

        </header>
        <div class="flex grow">
          <div class="h-full w-1/6 flex flex-col gap-2">
            <div v-if="mode === 0">
              <p>Mask Color</p>
              <a-color-picker v-model="color" disabledAlpha showText/>
            </div>
            <div v-if="mode === 1">
              <p>Path Suffix</p>
              <input v-model="mask_suffix" class="rounded border-gray-200 w-full" style="box-shadow: none;"
                     type="text"/>
            </div>
            <div v-if="mode === 2">
              <p>Select Detector</p>
              <select class="w-12 block p-2 border border-gray-300 bg-white rounded shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      name="detector">
              </select>
              <p>Configure Detector</p>
            </div>
            <div class="flex">
              <p class="mr-1">Is Mosaic</p>
              <input v-model="is_mosaic" class="rounded border-gray-200" style="box-shadow: none;"
                     type="checkbox"/>
            </div>
            <div class="w-full">
              <p class="mb-1">Output Directory</p>
              <input v-model="output" class="rounded border-gray-200 w-full"
                     style="box-shadow: none;" type="text"/>
            </div>
            <SubmitButton :disabled="submitDisabled" @click="() => waiting = true"/>
          </div>
          <div class="h-full w-2/3">CENTER IMAGE EDITOR</div>
          <div class="h-full w-1/6">
            <ViewFiles :files="getFilesJoined" @remove-file="removeFile"/>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
