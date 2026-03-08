<template>
  <details v-if="docs && docs.length" class="references">
    <summary>📚 参考文档 ({{ docs.length }})</summary>
    <div v-for="(doc, idx) in docs.slice(0,5)" :key="idx" class="doc-item">
      <p><strong>文档 {{ idx+1 }}</strong> 
        <span class="doc-similarity">相似度: {{ doc.similarity?.toFixed(4) }}</span>
      </p>
      <p>{{ doc.content_preview || doc.content.substring(0,150)+'...' }}</p>
    </div>
  </details>
</template>

<script setup>
defineProps({
  docs: {
    type: Array,
    default: () => []
  }
})
</script>