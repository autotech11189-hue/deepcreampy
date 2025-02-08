<template>
  <button :class="buttonClasses" @click="click">
    Submit
  </button>
</template>
<script lang="ts">
export default {
  props: {
    disabled: {
      type: Boolean,
      required: true
    },
  },
  computed: {
    buttonClasses() {
      return [
        "rounded p-2 font-bold",
        this.disabled
          ? "bg-gray-200 text-gray-400 cursor-not-allowed"
          : "bg-blue-500 hover:bg-blue-700 text-white cursor-pointer"
      ];
    },
  },
  methods: {
    click() {
      if (!this.disabled) {
        this.$emit("click");
      }
    },
  },
}
</script>

