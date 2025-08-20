import { useState, useRef, useCallback } from 'react';

export function useGamepad({
  socket,
  telloConnected,
  isConnected,
  isFlying,
  speed,
  setHumanDetection,
  setIsRecording,
  onSpeedChange,
}) {
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const [gamepadIndex, setGamepadIndex] = useState(-1);
  const [gamepadInfo, setGamepadInfo] = useState(null);
  const [gamepadInputs, setGamepadInputs] = useState({
    leftStick: { x: 0, y: 0 },
    rightStick: { x: 0, y: 0 },
    dpad: { up: false, down: false, left: false, right: false },
    buttons: {
      a: false, b: false, x: false, y: false,
      l1: false, r1: false, l2: 0, r2: 0,
      select: false, start: false, l3: false, r3: false, r4: false
    }
  });
  const [lastGamepadInputs, setLastGamepadInputs] = useState(gamepadInputs);

  // Detect gamepads
  const detectGamepads = useCallback(() => {
    const gamepads = navigator.getGamepads();
    let foundGamepad = false;
    for (let i = 0; i < gamepads.length; i++) {
      const gamepad = gamepads[i];
      if (gamepad) {
        if (gamepad.id.includes('SHANWAN') || gamepad.id.includes('Android Gamepad') || gamepad.id.includes('2563') || gamepad.id.includes('526')) {
          setGamepadConnected(true);
          setGamepadIndex(i);
          setGamepadInfo({ id: gamepad.id, index: i, buttons: gamepad.buttons.length, axes: gamepad.axes.length });
          foundGamepad = true;
          break;
        } else if (!foundGamepad) {
          setGamepadConnected(true);
          setGamepadIndex(i);
          setGamepadInfo({ id: gamepad.id, index: i, buttons: gamepad.buttons.length, axes: gamepad.axes.length });
          foundGamepad = true;
        }
      }
    }
    if (!foundGamepad && gamepadConnected) {
      setGamepadConnected(false);
      setGamepadIndex(-1);
      setGamepadInfo(null);
    }
  }, [gamepadConnected]);

  // Parse gamepad input
  const parseGamepadInput = useCallback((gamepad) => {
    if (!gamepad) return gamepadInputs;
    const leftStickX = gamepad.axes[0] || 0;
    const leftStickY = gamepad.axes[1] || 0;
    const rightStickX = gamepad.axes[2] || 0;
    const rightStickY = gamepad.axes[3] || 0;
    const buttons = {
      a: gamepad.buttons[0]?.pressed || false,
      b: gamepad.buttons[1]?.pressed || false,
      x: gamepad.buttons[3]?.pressed || false,
      y: gamepad.buttons[4]?.pressed || false,
      l1: gamepad.buttons[6]?.pressed || false,
      r1: gamepad.buttons[7]?.pressed || false,
      l2: gamepad.buttons[8]?.value || 0,
      r2: gamepad.buttons[9]?.value || 0,
      select: gamepad.buttons[10]?.pressed || false,
      start: gamepad.buttons[11]?.pressed || false,
      r3: gamepad.buttons[13]?.pressed || false,
      r4: gamepad.buttons[14]?.pressed || false
    };
    let dpad = { up: false, down: false, left: false, right: false };
    if (gamepad.axes.length > 6) {
      const dpadX = gamepad.axes[6] || 0;
      const dpadY = gamepad.axes[7] || 0;
      dpad.left = dpadX < -0.5;
      dpad.right = dpadX > 0.5;
      dpad.up = dpadY < -0.5;
      dpad.down = dpadY > 0.5;
    }
    return {
      leftStick: { x: leftStickX, y: leftStickY },
      rightStick: { x: rightStickX, y: rightStickY },
      dpad,
      buttons
    };
  }, [gamepadInputs]);

  // Process gamepad input
  const processGamepadInput = useCallback((currentInputs, previousInputs) => {
    if (!socket || !telloConnected || !isConnected) return;
    const curr = currentInputs.buttons;
    const prev = previousInputs.buttons;
    if (curr.a && !prev.a && !isFlying) {
      socket.emit('takeoff');
    }
    if (curr.b && !prev.b && isFlying) {
      socket.emit('land');
    }
    if (curr.start && !prev.start) {
      socket.emit('stop_movement');
      if (isFlying) socket.emit('emergency_land');
    }
    if (curr.l2 && !prev.l2) {
      setHumanDetection(prev => !prev);
    }
    if (currentInputs.dpad.up && !previousInputs.dpad.up && isFlying) {
      socket.emit('flip_command', { direction: 'f' });
    }
    if (currentInputs.dpad.down && !previousInputs.dpad.down && isFlying) {
      socket.emit('flip_command', { direction: 'b' });
    }
    if (currentInputs.dpad.left && !previousInputs.dpad.left && isFlying) {
      socket.emit('flip_command', { direction: 'l' });
    }
    if (currentInputs.dpad.right && !previousInputs.dpad.right && isFlying) {
      socket.emit('flip_command', { direction: 'r' });
    }
    if (curr.x && !prev.x) {
      socket.emit('manual_screenshot');
    }
    if (curr.y && !prev.y) {
      const newRecordingState = !isFlying;
      socket.emit('toggle_recording', { recording: newRecordingState });
      setIsRecording(newRecordingState);
    }
    if (curr.r4 && !prev.r4) {
      onSpeedChange(Math.min(speed + 10, 100));
    }
    if (curr.r3 && !prev.r3) {
      onSpeedChange(Math.max(speed - 10, 10));
    }
    if (isFlying) {
      const moveSpeed = Math.min(Math.max(speed, 10), 100);
      const deadzone = 0.15;
      let controls = { left_right: 0, for_back: 0, up_down: 0, yaw: 0 };
      if (Math.abs(currentInputs.leftStick.x) > deadzone) controls.left_right = Math.round(currentInputs.leftStick.x * moveSpeed);
      if (Math.abs(currentInputs.leftStick.y) > deadzone) controls.for_back = Math.round(-currentInputs.leftStick.y * moveSpeed);
      if (Math.abs(currentInputs.rightStick.y) > deadzone) controls.up_down = Math.round(-currentInputs.rightStick.y * moveSpeed);
      if (Math.abs(currentInputs.rightStick.x) > deadzone) controls.yaw = Math.round(currentInputs.rightStick.x * moveSpeed);
      socket.emit('move_control', controls);
    }
  }, [socket, telloConnected, isConnected, isFlying, speed, setHumanDetection, setIsRecording, onSpeedChange]);

  // Poll gamepad
  const pollGamepad = useCallback((controlMode) => {
    if (!gamepadConnected || gamepadIndex === -1) return;
    const gamepads = navigator.getGamepads();
    const gamepad = gamepads[gamepadIndex];
    if (gamepad) {
      const currentInputs = parseGamepadInput(gamepad);
      if (controlMode === 'Controller Mode') {
        processGamepadInput(currentInputs, lastGamepadInputs);
      }
      setGamepadInputs(currentInputs);
      setLastGamepadInputs(currentInputs);
    }
  }, [gamepadConnected, gamepadIndex, parseGamepadInput, processGamepadInput, lastGamepadInputs]);

  return {
    gamepadConnected,
    gamepadIndex,
    gamepadInfo,
    gamepadInputs,
    lastGamepadInputs,
    detectGamepads,
    parseGamepadInput,
    processGamepadInput,
    pollGamepad,
    setGamepadInputs,
    setLastGamepadInputs,
  };
}
