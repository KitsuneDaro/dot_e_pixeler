import Explain from "./Explain";
import Unit from "./Unit";

function SiteExplain() {
    return <Unit className="flex row">
        <div className="flex column text">
            <Explain>使い方</Explain>
            <Explain>
                <p>・このサイトでは画像をドット絵風に変換することができます。</p>
                <p>・「高速な変換」では各色がカバーする元の色の範囲を同じ位の広さに留めて変換します。</p>
                <p>　色が近いが区別される色(例えば白とパールオレンジ)などを1色で表してしまうことがあります。</p>
                <p>・「色を偏らせる変換」では色数などから判断して、色は近くても区別される色を別の色で表しやすくなっています。</p>
                <p>　ただし、その分他の色が削減されることもあり、何度か変換を試さないと良い画像が得られない場合があります。</p>
                <p>　また、計算速度が遅く、画像サイズが大きい場合や変換後の色数が多い場合にはより遅くなる場合があります。</p>
                <p>・また、変換に失敗して黒い画像が出力されることがあります。</p>
                <p>　発生条件は不明ですが、もう一度変換を行うことで正常に出力されることが多いです。</p>
                <p>・変換した画像を保存する場合は「保存」ボタンを押すか、あるいは変換後の画像を選択して通常の方法で保存することができます。</p>
            </Explain>
        </div>
        <div className="flex column text">
            <Explain>注意点</Explain>
            <Explain>
                <p>・変換後の画像を共有する場合、画像の一次作成者から許可をもらってください。</p>
                <p>　画像の一次作成者が不明、あるいは許可が出なかった場合には、共有せず個人での利用に留めてください。</p>
                <p>・元画像の取得元が転載である場合、その転載者は画像の一次作成者とはみなされません。</p>
                <p>　そのため、許可を得る場合にはその取得元が転載であるかどうかも確認してください。</p>
                <p>・その他、マナーを意識して利用してください。このサイトで被った不利益については一切責任を持ちません。</p>
            </Explain>
        </div>
    </Unit>;
}

export default SiteExplain