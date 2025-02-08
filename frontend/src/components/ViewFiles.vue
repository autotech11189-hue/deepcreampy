<template>
  <div class="h-full w-full overflow-auto">
    <ul>
      <li v-for="(file, index) in files" :key="index" class="flex align-center justify-between p-1">
        <span class="grow cursor-pointer" @click="viewFile(file)">{{ file.image.name }}</span>

        <span v-if="file.warn" class="color-warning-500 cursor-default">⚠️</span>

        <span class="font-bold color-danger-500 cursor-pointer" @click="removeFile(file)">❌</span>
      </li>
    </ul>
  </div>
</template>

<script>
export default {
  props: {
    files: {
      type: Array,
      required: true
    }
  },
  methods: {
    viewFile(file) {
      alert("No preview available");
    },
    removeFile(file) {
      this.$emit("remove-file", file);
    }
  }
};
</script>
