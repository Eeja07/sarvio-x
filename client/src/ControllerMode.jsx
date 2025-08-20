import React from 'react';

function ControllerMode(props) {
  const { leftColumnTextsController, rightColumnTextsController } = props;
  return (
    <div className="flex flex-col lg:flex-row w-full bg-deep-teal rounded-lg p-6 mb-2 gap-4">
      <div className="flex-1 w-full space-y-1">
        <h3 className="text-[15px] font-semibold text-ivory mb-2 border-b border-ivory/30 pb-1">Actions</h3>
        {leftColumnTextsController.map((text, idx) => (
          <div key={`left-${idx}`} className="flex items-center gap-2 py-1">
            <div className="w-20 h-5 bg-ivory text-deep-teal rounded flex items-center justify-center text-center font-bold text-[10px] shrink-0">{text.split(' = ')[1]}</div>
            <span className="text-[13px] text-ivory whitespace-nowrap">{text.split(' = ')[0] || text}</span>
          </div>
        ))}
      </div>
      <div className="flex-1 w-full space-y-1">
        <h3 className="text-[15px] font-semibold text-ivory mb-2 border-b border-ivory/30 pb-1">Movement</h3>
        {rightColumnTextsController.map((text, idx) => (
          <div key={`right-${idx}`} className="flex items-center gap-2 py-1">
            <div className={`w-20 h-5 bg-ivory text-deep-teal rounded flex items-center justify-center font-bold shrink-0 ${idx < 4 ? 'text-[10px]' : 'text-[10px]'}`}>{text.split(' = ')[1]}</div>
            <span className="text-[13px] text-ivory whitespace-nowrap">{text.split(' = ')[0]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ControllerMode;
