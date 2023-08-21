//整数型の入力
function IntInput({onChange, id, value, max = Infinity, min = 1}: {onChange: Function, id?: string, value?: number, max?: number, min?: number}) {
    return <input id={id} value={value} type="number"
    onChange={(event) => {
        onChange(event);
    }}
    onBlur={(event) => {
        event.target.value = String(Math.min(Math.max(parseInt(event.target.value), min), max));
    }} min={min} max={max}></input>;
};

export default IntInput