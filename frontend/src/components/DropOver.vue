<template>
  <div class="w-full h-full flex items-center justify-center">
  <div
    class="cursor-pointer w-[66%] h-[50%] border-1 border-dashed border-gray-400 flex items-center justify-center text-gray-500 rounded-lg"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
    @click="openFileInput"
    :class="{ 'bg-gray-100': isDragging }"
  >
    <p class="font-bold">Drag & Drop files here or click to upload</p>
    <input
      type="file"
      ref="fileInput"
      class="hidden"
      multiple
      @change="handleFileSelect"
    />
  </div>
    </div>
</template>

<script>
export default {
  data() {
    return {
      isDragging: false,
      files: []
    };
  },
  methods: {
    handleDrop(event) {
      this.isDragging = false;
      const droppedFiles = Array.from(event.dataTransfer.files);
      this.files.push(...droppedFiles);
      this.$emit('files-selected', this.files);
    },
    openFileInput() {
      this.$refs.fileInput.click();
    },
    handleFileSelect(event) {
      const selectedFiles = Array.from(event.target.files);
      this.files.push(...selectedFiles);
      this.$emit('files-selected', this.files);
    }
  }
};
</script>
