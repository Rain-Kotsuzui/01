import React from 'react';

function GameOverlay({ gameState, onStart, onRestart }) {
  if (gameState === 'start') {
    return (
      <div className="game-overlay">
        <h2>游戏说明</h2>
        <p>使用【方向键】控制蛇的移动。</p>
        <p>吃掉屏幕上的汉字，集满八个字，或按【F】键，即可开始作诗。</p>
        <button onClick={onStart}>开始游戏</button>
      </div>
    );
  }

  if (gameState === 'gameOver') {
    return (
      <div className="game-overlay">
        <h2>游戏结束</h2>
        <button onClick={onRestart}>重新开始</button>
      </div>
    );
  }

  return null;
}

export default GameOverlay;