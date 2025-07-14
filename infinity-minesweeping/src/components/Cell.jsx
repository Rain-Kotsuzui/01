// src/components/Cell.jsx
import React from 'react';

const Cell = ({ data, onClick, onContextMenu }) => {
  const renderContent = () => {
    if (data.state === 'FLAGGED') return '🚩';
    if (data.state === 'HIDDEN') return '';

    // state is REVEALED
    if (data.isMine) return '💣';
    if (data.adjacentMines > 0) {
      return data.adjacentMines;
    }
    return '';
  };

  const getClassName = () => {
    let className = 'cell';
    if (data.state === 'REVEALED') {
      className += ' revealed';
      if (!data.isMine) {
        className += ` number-${data.adjacentMines}`;
      } else {
        className += ' mine';
      }
    }
    if (data.state === 'FLAGGED') {
        className += ' flagged';
    }
    return className;
  };

  return (
    <div
      className={getClassName()}
      onClick={onClick}
      onContextMenu={onContextMenu}
    >
      {renderContent()}
    </div>
  );
};

// 使用 React.memo 优化性能
export default React.memo(Cell);