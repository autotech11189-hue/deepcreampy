<template>
  <button
    :class="buttonClasses"
    :disabled="disabled"
    @click="toggleActive"
  >
    {{ content }}
  </button>
</template>

<script>
export default {
  props: {
    left: {
      type: Boolean,
      required: false
    },
    right: {
      type: Boolean,
      required: false
    },
    content: {
      type: String,
      required: true,
    },
    disabled: {
      type: Boolean,
      default: false,
    },
    active: {
      type: Boolean,
      default: true,
    },
  },
  computed: {
    buttonClasses() {
      return [
        "font-bold py-2 px-4 transition-all duration-300",
        "focus:outline-none",
        this.left ? "rounded-l" : "",
        this.right ? "rounded-r" : "",
        this.disabled
          ? "bg-gray-200 text-gray-400 cursor-not-allowed"
          : this.active
            ? "bg-blue-500 hover:bg-blue-700 text-white"
            : "bg-gray-300 hover:bg-gray-400 text-gray-800",
      ];
    },
  },
  methods: {
    toggleActive() {
      if (!this.disabled) {
        this.$emit("update:active");
      }
    },
  },
};
</script>
