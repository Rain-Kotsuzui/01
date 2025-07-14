import React from 'react';

function GameInfo({ apiKey, onApiKeyChange, collectedChars }) {
  return (
    <div className="info-panel">
      <div className="api-key-section">
        <label htmlFor="apiKey">DeepSeek API Key (安全提示：请勿在公共场合暴露Key)</label>
        <input 
          type="password" 
          id="apiKey" 
          placeholder="请在此处粘贴您的DeepSeek API Key"
          value={apiKey}
          onChange={onApiKeyChange}
        />
      </div>
      <div className="collected-chars-section">
        <h3>已拾得的汉字：</h3>
        <p id="collected-chars-display">
          {collectedChars.length > 0 ? collectedChars.join(' ') : '【等待拾取...】'}
        </p>
      </div>
    </div>
  );
}

export default GameInfo;