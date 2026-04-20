// ===== Copy BibTeX =====
document.addEventListener('DOMContentLoaded', () => {
  const copyBtn = document.getElementById('bibtex-copy-btn');
  if (!copyBtn) return;

  copyBtn.addEventListener('click', () => {
    const bibtexBlock = document.getElementById('bibtex-text');
    const text = bibtexBlock.textContent;

    navigator.clipboard.writeText(text).then(() => {
      copyBtn.textContent = 'Copied!';
      copyBtn.classList.add('copied');
      setTimeout(() => {
        copyBtn.textContent = 'Copy';
        copyBtn.classList.remove('copied');
      }, 2000);
    }).catch(() => {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      copyBtn.textContent = 'Copied!';
      copyBtn.classList.add('copied');
      setTimeout(() => {
        copyBtn.textContent = 'Copy';
        copyBtn.classList.remove('copied');
      }, 2000);
    });
  });
});
