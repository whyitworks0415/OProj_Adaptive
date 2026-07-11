// ReplayCameraBenchmark.cs
// VulkanRenderer 에서 녹화한 카메라 경로(replays/replay_NNN.csv)를 Unity 에서
// "완전히 동일한 움직임"으로 재생하면서 FPS/프레임타임을 CSV 로 기록한다.
// -> 같은 맵 + 같은 움직임에서 자체 Vulkan 렌더러 vs Unity 성능 비교 실험용.
//
// 사용법:
//  1. 맵 GLB 를 Unity 로 임포트 (glTFast 패키지 권장: Window > Package Manager)
//  2. Main Camera 에 이 스크립트를 붙인다.
//  3. replay_NNN.csv 를 Assets/StreamingAssets/ 에 복사하고 csvFileName 에 파일명 입력
//     (또는 csvAsset 에 TextAsset 으로 드래그)
//  4. Play -> 경로 재생이 끝나면 결과 CSV 경로가 Console 에 출력된다.
//
// 좌표계: 렌더러는 오른손(Y-up), Unity 는 왼손(Y-up)이라 축 하나를 반전해야 한다.
//  - glTFast 로 맵을 임포트했으면 FlipX (기본값)
//  - 맵이 좌우로 뒤집혀 보이면 FlipZ 나 None 으로 바꿔 볼 것
//
// 공정 비교 체크리스트:
//  - 해상도 동일 (렌더러 창모드 1280x720 이면 Unity Game 뷰도 1280x720 고정)
//  - VSync OFF (이 스크립트가 기본으로 꺼 줌), FPS 캡 동일 (렌더러 I 키와 targetFps 맞추기)
//  - 에디터 Play 모드보다 Standalone 빌드가 훨씬 공정하다 (에디터 오버헤드 큼)

using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

public class ReplayCameraBenchmark : MonoBehaviour
{
    public enum AxisConversion { FlipX, FlipZ, None }

    [Header("입력 (둘 중 하나)")]
    [Tooltip("Assets/StreamingAssets/ 안의 CSV 파일명 (예: replay_001.csv)")]
    public string csvFileName = "replay_001.csv";
    [Tooltip("또는 CSV 를 TextAsset 으로 직접 지정 (이 값이 있으면 우선)")]
    public TextAsset csvAsset;

    [Header("좌표계 변환")]
    [Tooltip("glTFast 로 맵을 임포트했으면 FlipX. 맵이 반전돼 보이면 바꿔 볼 것.")]
    public AxisConversion axisConversion = AxisConversion.FlipX;

    [Header("측정 설정")]
    [Tooltip("측정 시작 전 워밍업 시간 (초) — 셰이더 컴파일/로딩 스파이크 제외")]
    public float warmupSeconds = 1.0f;
    [Tooltip("0 = 무제한(최대 FPS 모드). 렌더러의 I 키 캡과 맞출 것 (예: 60)")]
    public int targetFps = 0;
    public bool disableVSync = true;
    [Tooltip("재생이 끝나면 자동으로 Play 종료 (Standalone 이면 앱 종료)")]
    public bool quitOnFinish = false;

    struct PathFrame { public float time; public Vector3 pos; public float yaw, pitch; }
    struct Sample    { public float time, fps, ftMs; }

    readonly List<PathFrame> frames  = new List<PathFrame>();
    readonly List<Sample>    samples = new List<Sample>();
    int   frameIdx  = 0;
    float playTime  = -1f;   // <0 이면 아직 시작 안 함
    bool  finished  = false;

    void Start()
    {
        if (disableVSync) QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = (targetFps > 0) ? targetFps : -1;

        string text = null;
        if (csvAsset != null) text = csvAsset.text;
        else
        {
            string path = Path.Combine(Application.streamingAssetsPath, csvFileName);
            if (File.Exists(path)) text = File.ReadAllText(path);
            else { Debug.LogError($"[ReplayBench] CSV not found: {path}"); enabled = false; return; }
        }
        Parse(text);
        if (frames.Count < 2)
        {
            Debug.LogError("[ReplayBench] CSV 에 유효한 프레임이 2개 미만입니다.");
            enabled = false; return;
        }
        Debug.Log($"[ReplayBench] Loaded {frames.Count} frames, " +
                  $"duration {frames[frames.Count - 1].time:F1}s, conversion={axisConversion}");
        playTime = 0f;
        Apply(frames[0]); // 시작 위치로 즉시 이동
    }

    void Parse(string text)
    {
        var inv = CultureInfo.InvariantCulture;
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.Trim();
            if (line.Length == 0 || line[0] == '#' || line.StartsWith("time_s")) continue;
            string[] c = line.Split(',');
            if (c.Length < 6) continue;
            PathFrame f;
            f.time  = float.Parse(c[0], inv);
            f.pos   = new Vector3(float.Parse(c[1], inv), float.Parse(c[2], inv), float.Parse(c[3], inv));
            f.yaw   = float.Parse(c[4], inv);
            f.pitch = float.Parse(c[5], inv);
            frames.Add(f);
        }
    }

    // 렌더러와 동일한 공식으로 전방 벡터를 만든 뒤 좌표계 변환을 적용한다.
    // front = (cosP*cosY, sinP, cosP*sinY)  — 오른손, Y-up, 각도는 도(degree)
    Vector3 Forward(float yawDeg, float pitchDeg)
    {
        float y = yawDeg * Mathf.Deg2Rad, p = pitchDeg * Mathf.Deg2Rad;
        var d = new Vector3(Mathf.Cos(p) * Mathf.Cos(y), Mathf.Sin(p), Mathf.Cos(p) * Mathf.Sin(y));
        return Convert(d);
    }

    Vector3 Convert(Vector3 v)
    {
        switch (axisConversion)
        {
            case AxisConversion.FlipX: return new Vector3(-v.x, v.y, v.z);
            case AxisConversion.FlipZ: return new Vector3(v.x, v.y, -v.z);
            default:                   return v;
        }
    }

    void Apply(PathFrame f)
    {
        transform.position = Convert(f.pos);
        transform.rotation = Quaternion.LookRotation(Forward(f.yaw, f.pitch), Vector3.up);
    }

    void Update()
    {
        if (finished || playTime < 0f) return;
        playTime += Time.unscaledDeltaTime;

        // 렌더러의 재생 로직과 동일: 시간 기반 인덱스 진행 + 인접 프레임 선형 보간
        while (frameIdx + 1 < frames.Count && frames[frameIdx + 1].time <= playTime)
            ++frameIdx;

        if (frameIdx + 1 < frames.Count)
        {
            PathFrame f0 = frames[frameIdx], f1 = frames[frameIdx + 1];
            float span = f1.time - f0.time;
            float t = (span > 0f) ? Mathf.Clamp01((playTime - f0.time) / span) : 1f;
            PathFrame f;
            f.time  = playTime;
            f.pos   = Vector3.Lerp(f0.pos, f1.pos, t); // 변환 전 원좌표로 보간 (렌더러와 동일)
            f.yaw   = f0.yaw   + t * (f1.yaw   - f0.yaw);
            f.pitch = f0.pitch + t * (f1.pitch - f0.pitch);
            Apply(f);
        }
        else
        {
            Apply(frames[frames.Count - 1]);
            Finish();
            return;
        }

        // 워밍업 이후부터 측정
        float dt = Time.unscaledDeltaTime;
        if (playTime > warmupSeconds && dt > 0f && dt < 0.25f)
            samples.Add(new Sample { time = playTime, fps = 1f / dt, ftMs = dt * 1000f });
    }

    void Finish()
    {
        finished = true;
        if (samples.Count == 0) { Debug.LogWarning("[ReplayBench] No samples."); return; }

        float sumFt = 0f, minFt = float.MaxValue, maxFt = 0f;
        foreach (var s in samples)
        {
            sumFt += s.ftMs;
            if (s.ftMs < minFt) minFt = s.ftMs;
            if (s.ftMs > maxFt) maxFt = s.ftMs;
        }
        float avgFt  = sumFt / samples.Count;
        float avgFps = (avgFt > 0f) ? 1000f / avgFt : 0f;

        var sb = new StringBuilder();
        sb.AppendLine("frame,time_s,fps,frametime_ms");
        var inv = CultureInfo.InvariantCulture;
        for (int i = 0; i < samples.Count; ++i)
            sb.AppendLine(string.Format(inv, "{0},{1:F4},{2:F2},{3:F4}",
                                        i, samples[i].time, samples[i].fps, samples[i].ftMs));
        sb.AppendLine();
        sb.AppendLine("# summary: avg_fps,avg_ft_ms,min_ft_ms,max_ft_ms,samples");
        sb.AppendLine(string.Format(inv, "summary,{0:F2},{1:F4},{2:F4},{3:F4},{4}",
                                    avgFps, avgFt, minFt, maxFt, samples.Count));
        sb.AppendLine($"# replay_csv: {(csvAsset ? csvAsset.name : csvFileName)}  conversion: {axisConversion}");
        sb.AppendLine($"# resolution: {Screen.width}x{Screen.height}  vsync: {QualitySettings.vSyncCount}  target_fps: {targetFps}");
        sb.AppendLine($"# unity: {Application.unityVersion}  quality: {QualitySettings.names[QualitySettings.GetQualityLevel()]}");

        string outPath = Path.Combine(Application.persistentDataPath,
            $"unity_bench_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv");
        File.WriteAllText(outPath, sb.ToString());
        Debug.Log($"[ReplayBench] Done!  avg FPS {avgFps:F1}  avg ft {avgFt:F2} ms\nSaved: {outPath}");

        if (quitOnFinish)
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
}
