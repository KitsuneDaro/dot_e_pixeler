import { ReactNode } from 'react';
import './Explain.css';

//整数型の入力
function Explain({children}: {children: ReactNode}) {
    return <div className='explain'>{children}</div>;
};

export default Explain