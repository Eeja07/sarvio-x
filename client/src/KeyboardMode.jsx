import React from 'react';

function KeyboardMode(props) {
  const { leftColumnTextsKeyboard, rightColumnTextsKeyboard } = props;
  return (
    <div className="flex flex-col lg:flex-row w-full bg-deep-teal rounded-lg p-9 mb-2 gap-6">
      <div className="flex-1 w-full space-y-1">
        <h3 className="text-[15px] font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">Actions</h3>
        {leftColumnTextsKeyboard.map((text, idx) => (
          <div key={`left-${idx}`} className="flex items-center space-x-3">
            <div className="w-8 h-6 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-[12px]">{text.split(' = ')[1]}</div>
            <span className="text-[13px] text-ivory">{text.split(' = ')[0] || text}</span>
          </div>
        ))}
      </div>
      <div className="flex-1 w-full space-y-1">
        <h3 className="text-[15px] font-semibold text-ivory mb-4 border-b border-ivory/30 pb-2">Movement</h3>
        {rightColumnTextsKeyboard.map((text, idx) => (
          <div key={`right-${idx}`} className="flex items-center space-x-3">
            <div className="w-8 h-6 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold text-[12px]">{text.includes('ARROW UP') ? '↑' : text.includes('ARROW DOWN') ? '↓' : text.includes('ARROW LEFT') ? '←' : text.includes('ARROW RIGHT') ? '→' : text.split(' = ')[1]}</div>
            <span className="text-[13px] text-ivory">{text.split(' = ')[0]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default KeyboardMode;
