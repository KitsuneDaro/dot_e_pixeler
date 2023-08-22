import Parameter from './UI/Parameter';

import DrawImg from './UI/DrawImg';
import Setting from './UI/Setting';

import './App.css'
import './UI/Canvas.css';
import Unit from './UI/Unit';
import Explain from './UI/Explain';
import Loading from './UI/Loading';
import SiteExplain from './UI/SiteExplain';

function App() {
    const parameter = new Parameter();

    return (
        <>
            <div className='flex column'>
                <Unit>
                    <Explain>
                        ドット絵ぴくせらあ(PC推奨)
                    </Explain>
                </Unit>
                <DrawImg param={parameter}></DrawImg>
                <Setting param={parameter}></Setting>
                <SiteExplain></SiteExplain>
            </div>
            <Loading id='loading' param={parameter}></Loading>
        </>// この後に設定が入る
        // 設定
        // .幅高さ
        // .拡大率
        // .色数
        // ..GMM-kmeans比率
    );
}

/*
    <div>
        <a href="https://vitejs.dev" target="_blank">
            <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
            <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
    </div>
    <h1>Vite + React</h1>
    <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
            count is {count}
        </button>
        <p>
            Edit <code>src/App.tsx</code> and save to test HMR
        </p>
    </div>
    <p className="read-the-docs">
        Click on the Vite and React logos to learn more
    </p>
*/

export default App
