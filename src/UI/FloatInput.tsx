//浮動小数点型の入力
function FloatInput({onChange, id, value, max, min = 1, step = 0.01}: {onChange: Function, id?: string, value?: number, max?: number, min?: number, step?: number}) {
    return <input id={id} value={value} type="number" onChange={(event) => {onChange(event)}} min={min} max={max} step={step}></input>;
};

export default FloatInput