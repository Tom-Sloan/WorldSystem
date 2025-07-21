import { useState } from 'react';
import { useControls, folder } from 'leva';

function Ground() {
  const [color, setColor] = useState('grey');
  const handleClick = () => {
    return
  };

  const { height, size, display } = useControls("Ground", () => ({
    'Ground Settings': folder({
      height: { value: -2, min: -10, max: 10, step: 0.1 },
      size: { value: 10, min: 1, max: 20, step: 1 },
      display: { value: false }
    }, { collapsed: true })
  }));

  if (!display) return null;

  return (
    <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[0, height, 0]} onClick={handleClick}>
      <planeGeometry args={[size, size]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
}

export default Ground;