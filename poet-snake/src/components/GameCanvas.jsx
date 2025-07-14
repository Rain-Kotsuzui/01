import React, { useRef, useEffect } from 'react';

// --- Game Constants ---
const GRID_SIZE = 20;
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 400;
const CHARACTER_POOL = [
    '春', '风', '花', '月', '夜', '江', '山', '水', '人', '思', '云', '天', '雨',
    '雪', '舟', '楼', '酒', '琴', '棋', '书', '画', '梦', '归', '愁', '情', '心',
    '光', '影', '声', '香', '色', '空', '星', '霜', '秋', '叶', '寒', '暖', '流'
];

function GameCanvas({ gameState, onCharEaten, onGameOver, onCompose }) {
    const canvasRef = useRef(null);
    // 使用useRef来存储游戏状态，避免不必要的组件重渲染
    const snakeRef = useRef([{ x: 10 * GRID_SIZE, y: 10 * GRID_SIZE }]);
    const directionRef = useRef({ x: 0, y: 0 });
    const foodRef = useRef(null);
    const gameLoopRef = useRef(null);

    const placeFood = () => {
        let foodX, foodY;
        const snake = snakeRef.current;
        do {
            foodX = Math.floor(Math.random() * (CANVAS_WIDTH / GRID_SIZE)) * GRID_SIZE;
            foodY = Math.floor(Math.random() * (CANVAS_HEIGHT / GRID_SIZE)) * GRID_SIZE;
        } while (snake.some(segment => segment.x === foodX && segment.y === foodY));

        const randomChar = CHARACTER_POOL[Math.floor(Math.random() * CHARACTER_POOL.length)];
        foodRef.current = { x: foodX, y: foodY, char: randomChar };
    };

    const resetGame = () => {
        snakeRef.current = [{ x: 10 * GRID_SIZE, y: 10 * GRID_SIZE }];
        directionRef.current = { x: 0, y: 0 };
        placeFood();
    };

    const draw = (ctx) => {
        ctx.fillStyle = '#e8f5e9';
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Draw snake
        const snake = snakeRef.current;
        ctx.fillStyle = '#4CAF50';
        snake.forEach(segment => {
            ctx.fillRect(segment.x, segment.y, GRID_SIZE, GRID_SIZE);
            ctx.strokeStyle = '#e8f5e9';
            ctx.strokeRect(segment.x, segment.y, GRID_SIZE, GRID_SIZE);
        });
        ctx.fillStyle = '#388E3C';
        ctx.fillRect(snake[0].x, snake[0].y, GRID_SIZE, GRID_SIZE);

        // Draw food
        const food = foodRef.current;
        if (food) {
            ctx.fillStyle = '#d32f2f';
            ctx.font = `bold ${GRID_SIZE}px 'KaiTi', serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(food.char, food.x + GRID_SIZE / 2, food.y + GRID_SIZE / 2);
        }
    };

    const gameLoop = () => {
        const snake = snakeRef.current;
        const direction = directionRef.current;
        const head = { x: snake[0].x + direction.x, y: snake[0].y + direction.y };
        if (direction.x === 0 && direction.y === 0) {
            return; // Do nothing until the player presses a key
        }
        if (
            head.x < 0 || head.x >= CANVAS_WIDTH ||
            head.y < 0 || head.y >= CANVAS_HEIGHT ||
            snake.some(segment => segment.x === head.x && segment.y === head.y)
        ) {
            onGameOver();
            return;
        }

        snake.unshift(head);

        const food = foodRef.current;
        if (head.x === food.x && head.y === food.y) {
            onCharEaten(food.char);
            placeFood();
        } else {
            snake.pop();
        }

        const ctx = canvasRef.current.getContext('2d');
        draw(ctx);
    };

    useEffect(() => {
        if (gameState === 'playing') {
            resetGame();
            gameLoopRef.current = setInterval(gameLoop, 120);
        } else {
            clearInterval(gameLoopRef.current);
            // Draw one last time when game ends to show final state
            if (canvasRef.current) {
                const ctx = canvasRef.current.getContext('2d');
                draw(ctx);
            }
        }
        return () => clearInterval(gameLoopRef.current);
    }, [gameState]);


    useEffect(() => {
        const handleKeyDown = (e) => {
            if (gameState !== 'playing') {
                if (e.key.toLowerCase() === 'f') {
                    e.preventDefault();
                    onCompose();
                }
                return;
            };
            e.preventDefault();

            let newDirection = directionRef.current;
            switch (e.key) {
                case 'ArrowUp':
                    if (directionRef.current.y === 0) newDirection = { x: 0, y: -GRID_SIZE };
                    break;
                case 'ArrowDown':
                    if (directionRef.current.y === 0) newDirection = { x: 0, y: GRID_SIZE };
                    break;
                case 'ArrowLeft':
                    if (directionRef.current.x === 0) newDirection = { x: -GRID_SIZE, y: 0 };
                    break;
                case 'ArrowRight':
                    if (directionRef.current.x === 0) newDirection = { x: GRID_SIZE, y: 0 };
                    break;
                case 'f':
                case 'F':
                    e.preventDefault();
                    onCompose();
                    break;
                default:
                    break;
            }
            directionRef.current = newDirection;
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [gameState, onCompose]);

    return (
        <div className="game-area">
            <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />
        </div>
    );
}

export default GameCanvas;