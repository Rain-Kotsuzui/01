import React from 'react';

function PoemDisplay({ poemState }) {
  if (!poemState.isComposing && !poemState.text) return null;

  return (
    <div className="poem-section">
      <h3>AI 诗作：</h3>
      {poemState.isComposing && (
        <div id="poem-loading">
          <p>灵感汇聚中，请稍候...</p>
          <div className="loader"></div>
        </div>
      )}
      {poemState.error && <pre className="poem-error">{poemState.error}</pre>}
      {poemState.text && <pre id="poem-output">{poemState.text}</pre>}
    </div>
  );
}

export default PoemDisplay;