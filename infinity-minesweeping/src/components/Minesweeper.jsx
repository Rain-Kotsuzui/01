// src/components/Minesweeper.jsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Cell from './Cell';
import { getCellInfo, revealCells, isMineAt } from '../utils/gameLogic';
import './Minesweeper.css';

// --- 常量定义 ---
const CELL_SIZE = 30;
const MIN_ZOOM = 0.3;
const MAX_ZOOM = 3.0;
const ZOOM_SENSITIVITY = 0.001;
const EASING_FACTOR = 0.1; // 缓动系数调回稍快一点，以获得更好的感觉
const INERTIA_FRICTION = 0.92; // 惯性滚动的摩擦力
const MIN_INERTIA_SPEED = 0.05;

const Minesweeper = () => {
    // --- 逻辑状态 (触发 React Re-render) ---
    const [seed, setSeed] = useState(() => Date.now());
    const [gameMap, setGameMap] = useState(() => new Map());
    const [gameState, setGameState] = useState('PLAYING');
    const [score, setScore] = useState(0);
    const [isFirstClick, setIsFirstClick] = useState(true);
    const [viewVersion, setViewVersion] = useState(0);

    // --- 视觉/交互状态 (使用 useRef 避免 re-render) ---
    const gridRef = useRef(null);
    const viewportRef = useRef({ x: 0, y: 0, zoom: 1 });
    const targetZoomRef = useRef(1);
    const targetPositionRef = useRef({ x: 0, y: 0 });
    const isPanningRef = useRef(false);
    const panStartRef = useRef({ x: 0, y: 0 });
    const animationFrameRef = useRef(null);
    const [windowSize, setWindowSize] = useState({ width: window.innerWidth, height: window.innerHeight });

    // 用于惯性滚动的 Ref
    const panVelocityRef = useRef({ x: 0, y: 0 });
    const lastPanTimeRef = useRef(0);
    const clickStartPositionRef = useRef({ x: 0, y: 0 });

    // 更新窗口尺寸
    useEffect(() => {
        const handleResize = () => setWindowSize({ width: window.innerWidth, height: window.innerHeight });
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // 核心：直接更新 DOM 的 transform 样式
    const updateTransform = useCallback(() => {
        if (gridRef.current) {
            const { x, y, zoom } = viewportRef.current;
            const { width, height } = windowSize;
            gridRef.current.style.transform = `translate(${width / 2 - x * zoom}px, ${height / 2 - y * zoom}px) scale(${zoom})`;
        }
    }, [windowSize]);

    // 统一的动画循环，处理所有缓动效果
    const animate = useCallback(() => {
        let isAnimating = false;
        
        // 1. 处理缩放缓动
        const currentZoom = viewportRef.current.zoom;
        const targetZoom = targetZoomRef.current;
        if (Math.abs(targetZoom - currentZoom) > 0.001) {
            isAnimating = true;
            viewportRef.current.zoom += (targetZoom - currentZoom) * EASING_FACTOR;
        } else {
            viewportRef.current.zoom = targetZoom;
        }

        // 2. 处理平移缓动与惯性
        const { x: velX, y: velY } = panVelocityRef.current;
        if (Math.abs(velX) > MIN_INERTIA_SPEED || Math.abs(velY) > MIN_INERTIA_SPEED) {
            isAnimating = true;
            viewportRef.current.x += velX;
            viewportRef.current.y += velY;
            targetPositionRef.current.x = viewportRef.current.x;
            targetPositionRef.current.y = viewportRef.current.y;
            panVelocityRef.current.x *= INERTIA_FRICTION;
            panVelocityRef.current.y *= INERTIA_FRICTION;
        } else {
            panVelocityRef.current = { x: 0, y: 0 };
            const { x: currentX, y: currentY } = viewportRef.current;
            const { x: targetX, y: targetY } = targetPositionRef.current;
            const dx = targetX - currentX;
            const dy = targetY - currentY;
            if (Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05) {
                isAnimating = true;
                viewportRef.current.x += dx * EASING_FACTOR;
                viewportRef.current.y += dy * EASING_FACTOR;
            } else {
                viewportRef.current.x = targetX;
                viewportRef.current.y = targetY;
            }
        }
        
        // 3. 更新DOM并决定是否继续动画
        if (isAnimating) {
            updateTransform();
            animationFrameRef.current = requestAnimationFrame(animate);
        } else {
            animationFrameRef.current = null;
            setViewVersion(v => v + 1);
        }
    }, [updateTransform]);
    
    // 启动动画的统一入口
    const startAnimation = () => {
        if (!animationFrameRef.current) {
            animationFrameRef.current = requestAnimationFrame(animate);
        }
    };

    // 处理滚轮事件 (缩放)
    const handleWheel = (e) => {
        e.preventDefault();
        if (gameState === 'GAME_OVER') return;

        const { deltaY, clientX, clientY } = e;
        const { width, height } = windowSize;
        const { x, y, zoom } = viewportRef.current;

        const mouseXWorld = (clientX - width / 2 + x * zoom) / zoom;
        const mouseYWorld = (clientY - height / 2 + y * zoom) / zoom;
        
        const newTargetZoom = targetZoomRef.current * (1 - deltaY * ZOOM_SENSITIVITY);
        targetZoomRef.current = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newTargetZoom));
        
        viewportRef.current.x = mouseXWorld - (clientX - width / 2) / targetZoomRef.current;
        viewportRef.current.y = mouseYWorld - (clientY - height / 2) / targetZoomRef.current;
        
        targetPositionRef.current.x = viewportRef.current.x;
        targetPositionRef.current.y = viewportRef.current.y;

        startAnimation();
    };

    // --- 修改: 按下鼠标 ---
    const handleMouseDown = (e) => {
        // e.button: 0=左键, 1=中键, 2=右键

        // 为左键单击记录起始位置，用于区分 "单击" 和 "无效的拖拽"
        if (e.button === 0) {
            clickStartPositionRef.current = { x: e.clientX, y: e.clientY };
            return;
        }

        // 只在中键按下时开始平移
        if (e.button === 1) {
            e.preventDefault(); // 阻止浏览器默认的中键滚动行为
            
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            
            targetPositionRef.current.x = viewportRef.current.x;
            targetPositionRef.current.y = viewportRef.current.y;

            isPanningRef.current = true;
            panStartRef.current = { x: e.clientX, y: e.clientY };
            panVelocityRef.current = { x: 0, y: 0 };
            lastPanTimeRef.current = performance.now();
            if (gridRef.current) gridRef.current.style.cursor = 'grabbing';
        }
    };

    // 移动鼠标 (无需修改，因为它依赖 isPanningRef)
    const handleMouseMove = (e) => {
        if (!isPanningRef.current) return;
        
        const { zoom } = viewportRef.current;
        const dx = e.clientX - panStartRef.current.x;
        const dy = e.clientY - panStartRef.current.y;
        
        const moveX = dx / zoom;
        const moveY = dy / zoom;

        viewportRef.current.x -= moveX;
        viewportRef.current.y -= moveY;
        
        targetPositionRef.current.x = viewportRef.current.x;
        targetPositionRef.current.y = viewportRef.current.y;
        
        panVelocityRef.current = { x: -moveX, y: -moveY }; 
        
        panStartRef.current = { x: e.clientX, y: e.clientY };
        lastPanTimeRef.current = performance.now();
        
        updateTransform();
    };

    // --- 修改: 松开鼠标 ---
    const handleMouseUp = (e) => {
        // 只有在平移状态下（即中键曾被按下）才执行
        if (isPanningRef.current) {
            isPanningRef.current = false;
            // 将光标恢复为可抓取状态（或默认状态，取决于你的CSS）
            if (gridRef.current) gridRef.current.style.cursor = 'grab';
            startAnimation(); // 启动惯性滚动
        }
    };

    // 处理点击格子
    const handleCellClick = (e, x, y) => {
        e.stopPropagation();
        
        // 这个检查仍然有用，可以防止用户在按下左键后意外移动鼠标导致的误点
        const { x: startX, y: startY } = clickStartPositionRef.current;
        if (Math.abs(e.clientX - startX) > 5 || Math.abs(e.clientY - startY) > 5) {
            return; // 移动超过阈值，判定为无效点击
        }

        if (gameState === 'GAME_OVER') return;

        let currentSeed = seed;
        if (isFirstClick) {
            let newSeed = seed;
            while (isMineAt(x, y, newSeed)) {
                newSeed = Date.now() + Math.random();
            }
            setSeed(newSeed);
            currentSeed = newSeed;
            setIsFirstClick(false);
        }

        const cellData = getCellInfo(x, y, currentSeed, gameMap);
        if (cellData.state === 'REVEALED' || cellData.state === 'FLAGGED') return;

        const result = revealCells(x, y, currentSeed, gameMap, score);

        setGameMap(result.newGameMap);
        // 假设 gameLogic 返回了正确的 scoreToAdd
        setScore(prevScore => prevScore + result.scoreToAdd);
        if (result.gameState === 'GAME_OVER') {
            setGameState('GAME_OVER');
        }
    };

    // 处理右键
    const handleContextMenu = (e, x, y) => {
        e.preventDefault();
        e.stopPropagation();
        if (gameState === 'GAME_OVER' || isFirstClick) return;

        const key = `${x},${y}`;
        const newGameMap = new Map(gameMap);
        const cellData = getCellInfo(x, y, seed, newGameMap);

        if (cellData.state === 'REVEALED') return;

        const newCellData = { ...cellData };
        newCellData.state = cellData.state === 'HIDDEN' ? 'FLAGGED' : 'HIDDEN';
        newGameMap.set(key, newCellData);
        setGameMap(newGameMap);
    };

    // 重启游戏
    const restartGame = () => {
        const newSeed = Date.now();
        setSeed(newSeed);
        setGameMap(new Map());
        setGameState('PLAYING');
        setScore(0);
        setIsFirstClick(true);
        viewportRef.current = { x: 0, y: 0, zoom: 1 };
        targetZoomRef.current = 1;
        targetPositionRef.current = { x: 0, y: 0 };
        panVelocityRef.current = { x: 0, y: 0 };
        updateTransform();
        setViewVersion(v => v + 1);
    };

    // 重置视角函数
    const resetView = useCallback(() => {
        targetPositionRef.current = { x: 0, y: 0 };
        targetZoomRef.current = 1;
        panVelocityRef.current = { x: 0, y: 0 };
        startAnimation();
    }, []);

    // 监听键盘 'R' 键
    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key.toLowerCase() === 'r' && e.target.tagName.toLowerCase() !== 'input') {
                e.preventDefault();
                resetView();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [resetView]);


    // 渲染格子的逻辑
    const renderCells = useCallback(() => {
        const cells = [];
        const { x, y, zoom } = viewportRef.current;
        const { width, height } = windowSize;
        
        const worldLeft = x - width / 2 / zoom;
        const worldRight = x + width / 2 / zoom;
        const worldTop = y - height / 2 / zoom;
        const worldBottom = y + height / 2 / zoom;

        const startX = Math.floor(worldLeft / CELL_SIZE) - 1;
        const endX = Math.ceil(worldRight / CELL_SIZE) + 1;
        const startY = Math.floor(worldTop / CELL_SIZE) - 1;
        const endY = Math.ceil(worldBottom / CELL_SIZE) + 1;

        for (let j = startY; j <= endY; j++) {
            for (let i = startX; i <= endX; i++) {
                const key = `${i},${j}`;
                const cellData = getCellInfo(i, j, seed, gameMap);
                cells.push(
                    <div
                        key={key}
                        className="cell-wrapper"
                        style={{
                            position: 'absolute',
                            left: `${i * CELL_SIZE}px`,
                            top: `${j * CELL_SIZE}px`,
                            width: `${CELL_SIZE}px`,
                            height: `${CELL_SIZE}px`,
                        }}
                        onClick={(e) => handleCellClick(e, i, j)}
                        onContextMenu={(e) => handleContextMenu(e, i, j)}
                    >
                        <Cell data={cellData} />
                    </div>
                );
            }
        }
        return cells;
    }, [seed, gameMap, viewVersion, windowSize]); // 优化依赖项

    // 初始设置和清理
    useEffect(() => {
        updateTransform();
        // 阻止中键点击时触发的默认右键菜单（某些浏览器/插件行为）
        const preventDefault = (e) => e.preventDefault();
        const gridEl = gridRef.current;
        if(gridEl) {
          gridEl.addEventListener('auxclick', preventDefault);
        }
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if(gridEl) {
              gridEl.removeEventListener('auxclick', preventDefault);
            }
        };
    }, [updateTransform]);

    return (
        <div className="game-container">
            <div className="ui-overlay">
                <h1>无限扫雷</h1>
                <div className="score">分数: {score}</div>
                {/* --- 修改: 更新UI提示 --- */}
                <div className="controls-info">中键拖拽 | 滚轮缩放 | R键重置</div>
            </div>
            
            {gameState === 'GAME_OVER' && (
                    <div className="game-over-message">
                        <h2>游戏结束!</h2>
                        <button onClick={restartGame}>重新开始</button>
                    </div>
                )}
            <div
                className="grid-container"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp} // 当鼠标离开区域时也停止平移
                onWheel={handleWheel}
            >
                <div ref={gridRef} className="grid">
                    {renderCells()}
                </div>
            </div>
        </div>
    );
};

export default Minesweeper;