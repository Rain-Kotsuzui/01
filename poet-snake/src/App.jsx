import React, { useState, useEffect, useCallback } from 'react';
import GameCanvas from './components/GameCanvas';
import GameInfo from './components/GameInfo';
import PoemDisplay from './components/PoemDisplay';
import GameOverlay from './components/GameOverlay';
import './App.css';

function App() {
  // 'start', 'playing', 'composing', 'gameOver'
  const [gameState, setGameState] = useState('start'); 
  const [apiKey, setApiKey] = useState('');
  const [collectedChars, setCollectedChars] = useState([]);
  const [poemState, setPoemState] = useState({
    text: '',
    isComposing: false,
    error: null,
  });

  const handleStartGame = () => {
    setGameState('playing');
  };

  const handleRestart = () => {
    setCollectedChars([]);
    setPoemState({ text: '', isComposing: false, error: null });
    setGameState('start');
  };

  const handleGameOver = () => {
    setGameState('gameOver');
  };

  const handleCharEaten = useCallback((char) => {
    setCollectedChars(prevChars => [...prevChars, char]);
  }, []);
  
  const handleComposePoem = useCallback(async () => {
    if (poemState.isComposing || collectedChars.length === 0) return;

    if (!apiKey.trim()) {
      alert('请输入您的DeepSeek API Key！');
      return;
    }
    
    setGameState('composing');
    setPoemState({ text: '', isComposing: true, error: null });

    const prompt = `请你扮演一位才华横溢的中国古代诗人，比如李白或杜甫。请使用我提供的汉字【${collectedChars.join('，')}】来创作一首意境优美、符合格律的中国古诗。诗歌的风格可以是五言或七言，绝句或律诗均可，但总长度不要超过八句话。请确保诗歌中至少包含大部分我提供的汉字。`;

    try {
      const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'deepseek-chat',
          messages: [
            { "role": "system", "content": "你是一位精通中国古典诗词的AI助手。" },
            { "role": "user", "content": prompt }
          ],
          max_tokens: 200,
          temperature: 0.8,
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API 请求失败: ${response.statusText} - ${errorData.error?.message || '未知错误'}`);
      }

      const data = await response.json();
      const poem = data.choices[0].message.content;
      setPoemState({ text: poem.trim(), isComposing: false, error: null });

    } catch (error) {
      console.error('作诗失败:', error);
      setPoemState({ 
        text: '', 
        isComposing: false, 
        error: `作诗失败，请检查API Key或网络连接。\n错误信息: ${error.message}`
      });
    } finally {
      handleGameOver();
    }
  }, [apiKey, collectedChars, poemState.isComposing]);
  
  // 检查是否集满8个字
  useEffect(() => {
    if (collectedChars.length >= 8 && gameState === 'playing') {
      handleComposePoem();
    }
  }, [collectedChars, gameState, handleComposePoem]);


  return (
    <div className="container">
      <h1>诗仙贪吃蛇</h1>
      
      <div className="game-area-wrapper">
        <GameOverlay 
          gameState={gameState} 
          onStart={handleStartGame} 
          onRestart={handleRestart}
        />
        <GameCanvas 
          gameState={gameState}
          onCharEaten={handleCharEaten}
          onGameOver={handleGameOver}
          onCompose={handleComposePoem}
        />
      </div>

      <GameInfo 
        apiKey={apiKey}
        onApiKeyChange={(e) => setApiKey(e.target.value)}
        collectedChars={collectedChars}
      />

      <PoemDisplay poemState={poemState} />
    </div>
  );
}

export default App;