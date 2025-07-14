// src/utils/gameLogic.js

// 使用一个简单的伪随机数生成器库，以保证在不同机器上结果一致
// 你可以 npm install seedrandom
import seedrandom from 'seedrandom';

const MINE_DENSITY = 0.16; // 16% 的雷密度

// 生成一个基于坐标和种子的确定性哈希值
function getCoordSeed(x, y, seed) {
  const prng = seedrandom(`${x},${y},${seed}`);
  return prng();
}

// 判断特定坐标是否是雷
export function isMineAt(x, y, seed) {
  return getCoordSeed(x, y, seed) < MINE_DENSITY;
}

// 获取或生成一个格子的信息（会使用缓存）
export function getCellInfo(x, y, seed, gameMap) {
  const key = `${x},${y}`;
  if (gameMap.has(key)) {
    return gameMap.get(key);
  }

  // 如果不在缓存中，则生成
  const isMine = isMineAt(x, y, seed);
  let adjacentMines = 0;
  if (!isMine) {
    // 遍历8个邻居计算雷数
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        if (dx === 0 && dy === 0) continue;
        if (isMineAt(x + dx, y + dy, seed)) {
          adjacentMines++;
        }
      }
    }
  }

  const cellData = {
    isMine,
    adjacentMines,
    state: 'HIDDEN', // HIDDEN, REVEALED, FLAGGED
  };
  
  return cellData;
}

// 揭开格子的核心逻辑（使用BFS进行 Flood Fill）
export function revealCells(x, y, seed, initialGameMap, initialScore) {
  const newGameMap = new Map(initialGameMap);
  let scoreToAdd = 0;
  
  const startCellKey = `${x},${y}`;
  const startCell = getCellInfo(x, y, seed, newGameMap);

  // 如果点击的是雷，游戏结束
  if (startCell.isMine) {
    newGameMap.set(startCellKey, { ...startCell, state: 'REVEALED' });
    return { newGameMap, scoreToAdd: 0, gameState: 'GAME_OVER' };
  }

  const queue = [[x, y]];
  const visited = new Set([startCellKey]);

  while (queue.length > 0) {
    const [cx, cy] = queue.shift();
    const key = `${cx},${cy}`;
    
    // 从缓存或重新生成获取格子信息
    const cell = getCellInfo(cx, cy, seed, newGameMap);

    // 如果已揭开，则跳过
    if (newGameMap.get(key)?.state === 'REVEALED') continue;
    
    // 揭开并计分
    newGameMap.set(key, { ...cell, state: 'REVEALED' });
    scoreToAdd += (cell.adjacentMines * 5);

    // 如果是0，将邻居加入队列
    if (cell.adjacentMines === 0) {
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          if (dx === 0 && dy === 0) continue;
          const nx = cx + dx;
          const ny = cy + dy;
          const neighborKey = `${nx},${ny}`;
          if (!visited.has(neighborKey)) {
            visited.add(neighborKey);
            queue.push([nx, ny]);
          }
        }
      }
    }
  }

  return { newGameMap, scoreToAdd, gameState: 'PLAYING' };
}