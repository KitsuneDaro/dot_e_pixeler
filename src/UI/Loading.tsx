import './Loading.css';
import Parameter from './Parameter';

function Loading({ id, param }: { id: string, param: Parameter }) {
    return <div className='loading' id={id} hidden={param.loadingHidden}>計算中……</div>;
}

export default Loading