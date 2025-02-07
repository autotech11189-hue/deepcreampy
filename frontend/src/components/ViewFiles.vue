<template>
  <div class="h-full w-full overflow-auto">
    <ul>
      <li v-for="(file, index) in files" :key="index" class="flex align-center justify-between p-1">
        <span @click="viewFile(file)" class="grow cursor-pointer">{{ file.image.name }}</span>

        <span v-if="file.warn" class="color-warning-500 cursor-default">⚠️</span>

        <span @click="removeFile(file)" class="font-bold color-danger-500 cursor-pointer">❌</span>
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
