export function SpecPage() {
  return (
    <div className="page">
      <section className="card specCard">
        <div className="specHeader">期末專題</div>
        <div className="specBody">
          <div className="specBlock">
            <div className="specTitle">題目：手寫數字辨識 或 語音數字辨識 擇一</div>
            <div className="specText">系統實作：需整合 AI 辨識模型，並設計系統介面，可即時展示</div>
          </div>

          <div className="specBlock">
            <div className="specTitle">繳交資料：</div>
            <ol className="specList">
              <li>程式碼（含註解）</li>
              <li>文字說明檔，說明系統架構、執行方式及執行流程</li>
              <li>系統執行之錄影檔，用以系統展示（如檔案太大，可提供連結）</li>
              <li>可使用語言模型輔助 AI 模型設計及系統設計</li>
            </ol>
          </div>

          <div className="specBlock">
            <div className="specTitle">評分標準：</div>
            <ol className="specList">
              <li>系統功能愈完整，分數愈高</li>
              <li>模型準確度愈高，分數愈高（要能展示或統計其準確度）</li>
            </ol>
          </div>

          <div className="specBlock">
            <div className="specTitle">本專案目前提供：</div>
            <ul className="specList">
              <li>手寫數字辨識：ONNX 模型 + 瀏覽器端推論</li>
              <li>語音數字辨識：瀏覽器內建語音辨識（若裝置支援）</li>
              <li>完整前端介面：可直接展示與錄影</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
