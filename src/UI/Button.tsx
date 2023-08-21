import React from 'react';
import './Button.css';

function Button({onClick = () => {}, innerText = '', style = {width: '4em', height: '2em', fontSize: '2em'}}: { onClick?: React.MouseEventHandler, innerText?: string, style?: {}}) {
    return <button onClick={onClick} style={style}>
        <div className='bottom'>
        </div>
        <div className="top">
            <span>{innerText}</span>
        </div>
    </button>
}

export default Button