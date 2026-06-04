import React from "react";
import {
  AbsoluteFill,
  Audio,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

export const FPS = 30;
export const INTRO_FRAMES = 120; // 4.0s
export const OUTRO_FRAMES = 180; // 6.0s
export const PLAYBACK = 1.9; // speed-up applied to captured footage (two prompts)
export const BGM = "bgm3.mp3";

const FONT = `${fontFamily}, -apple-system, "Segoe UI", Roboto, sans-serif`;
const ACCENT = "linear-gradient(110deg,#ec4899 0%,#8b5cf6 52%,#22d3ee 100%)";

// ---------- shared bits -----------------------------------------------------

const Stage: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill
    style={{
      fontFamily: FONT,
      background:
        "radial-gradient(1200px 700px at 50% 16%, #161226 0%, #0a0810 55%, #06060b 100%)",
      color: "#e7ecf3",
      overflow: "hidden",
    }}
  >
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(540px 380px at 14% 86%, rgba(236,72,153,0.18), transparent 70%), radial-gradient(560px 380px at 86% 14%, rgba(34,211,238,0.16), transparent 70%), radial-gradient(620px 420px at 60% 50%, rgba(139,92,246,0.12), transparent 72%)",
      }}
    />
    {children}
  </AbsoluteFill>
);

const GradientText: React.FC<{
  children: React.ReactNode;
  size: number;
  weight?: number;
  ls?: number;
}> = ({ children, size, weight = 800, ls = -1 }) => (
  <span
    style={{
      fontSize: size,
      fontWeight: weight,
      letterSpacing: ls,
      lineHeight: 1.05,
      backgroundImage: ACCENT,
      WebkitBackgroundClip: "text",
      backgroundClip: "text",
      color: "transparent",
    }}
  >
    {children}
  </span>
);

// ---------- intro -----------------------------------------------------------

const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pop = spring({ frame, fps, config: { damping: 11, mass: 0.7 } });
  const logoScale = interpolate(pop, [0, 1], [0.4, 1]);
  const logoRot = interpolate(pop, [0, 1], [-16, 0]);

  const rise = (start: number) => ({
    opacity: interpolate(frame, [start, start + 18], [0, 1], { extrapolateRight: "clamp" }),
    transform: `translateY(${interpolate(frame, [start, start + 18], [26, 0], {
      extrapolateRight: "clamp",
      easing: Easing.out(Easing.cubic),
    })}px)`,
  });

  const fadeOut = interpolate(frame, [INTRO_FRAMES - 16, INTRO_FRAMES], [1, 0], {
    extrapolateLeft: "clamp",
  });

  const steps = ["🗺️ Plan", "🌳 Expand", "🛠️ Refine", "📊 Score", "🧠 Remember"];

  return (
    <Stage>
      <AbsoluteFill
        style={{
          opacity: fadeOut,
          justifyContent: "center",
          alignItems: "center",
          gap: 32,
          padding: 80,
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 168,
            height: 168,
            borderRadius: 40,
            display: "grid",
            placeItems: "center",
            fontSize: 92,
            background: ACCENT,
            boxShadow: "0 30px 90px rgba(236,72,153,0.4)",
            transform: `scale(${logoScale}) rotate(${logoRot}deg)`,
          }}
        >
          🌳
        </div>

        <div style={rise(12)}>
          <GradientText size={118}>ReTreVal</GradientText>
        </div>

        <div style={{ ...rise(26), fontSize: 40, color: "#aeb8c8", fontWeight: 500 }}>
          A reasoning <b style={{ color: "#e7ecf3" }}>tree</b> that validates itself —{" "}
          <b style={{ color: "#e7ecf3" }}>and remembers</b>
        </div>

        <div style={{ ...rise(40), display: "flex", gap: 14, marginTop: 6 }}>
          {steps.map((m, i) => {
            const s = 46 + i * 6;
            const o = interpolate(frame, [s, s + 14], [0, 1], { extrapolateRight: "clamp" });
            const y = interpolate(frame, [s, s + 14], [16, 0], {
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            });
            return (
              <span
                key={m}
                style={{
                  opacity: o,
                  transform: `translateY(${y}px)`,
                  fontSize: 28,
                  fontWeight: 600,
                  padding: "11px 22px",
                  borderRadius: 999,
                  color: "#dbe4f0",
                  border: "1px solid rgba(140,120,200,0.4)",
                  background: "rgba(20,16,34,0.6)",
                }}
              >
                {m}
              </span>
            );
          })}
        </div>
      </AbsoluteFill>
    </Stage>
  );
};

// ---------- app footage -----------------------------------------------------

const AppClip: React.FC<{ durationInFrames: number }> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const fadeIn = interpolate(frame, [0, 14], [0, 1], { extrapolateRight: "clamp" });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 14, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp" },
  );
  return (
    <AbsoluteFill style={{ background: "#06060b", opacity: Math.min(fadeIn, fadeOut) }}>
      <OffthreadVideo
        src={staticFile("app.mp4")}
        playbackRate={PLAYBACK}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </AbsoluteFill>
  );
};

// ---------- outro -----------------------------------------------------------

const Stat: React.FC<{ value: string; label: string; delay: number }> = ({
  value,
  label,
  delay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 13 } });
  return (
    <div
      style={{
        opacity: s,
        transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)`,
        minWidth: 300,
        padding: "28px 34px",
        borderRadius: 24,
        background: "rgba(20,16,34,0.7)",
        border: "1px solid rgba(140,120,200,0.25)",
        textAlign: "center",
      }}
    >
      <div style={{ marginBottom: 8 }}>
        <GradientText size={64}>{value}</GradientText>
      </div>
      <div style={{ fontSize: 24, color: "#aeb8c8", fontWeight: 500 }}>{label}</div>
    </div>
  );
};

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const headO = interpolate(frame, [4, 24], [0, 1], { extrapolateRight: "clamp" });
  const headY = interpolate(frame, [4, 24], [24, 0], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const cta = spring({ frame: frame - 80, fps, config: { damping: 12 } });
  const linksO = interpolate(frame, [96, 116], [0, 1], { extrapolateRight: "clamp" });
  const fadeIn = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: "clamp" });

  return (
    <Stage>
      <AbsoluteFill
        style={{
          opacity: fadeIn,
          justifyContent: "center",
          alignItems: "center",
          gap: 44,
          padding: 90,
          textAlign: "center",
        }}
      >
        <div style={{ opacity: headO, transform: `translateY(${headY}px)` }}>
          <div style={{ fontSize: 80, fontWeight: 800, letterSpacing: -1 }}>
            Explore. Validate. <GradientText size={80}>Remember.</GradientText>
          </div>
        </div>

        <div style={{ display: "flex", gap: 24 }}>
          <Stat value="85.8%" label="accuracy on MATH-500" delay={26} />
          <Stat value="0" label="complete failures" delay={38} />
          <Stat value="4" label="LLM backends" delay={50} />
        </div>

        <div
          style={{
            transform: `scale(${interpolate(cta, [0, 1], [0.8, 1])})`,
            opacity: cta,
            fontSize: 34,
            fontWeight: 700,
            color: "#1a0820",
            padding: "20px 44px",
            borderRadius: 999,
            background: ACCENT,
            boxShadow: "0 18px 60px rgba(236,72,153,0.45)",
          }}
        >
          ★ Star us on GitHub
        </div>

        <div style={{ opacity: linksO, fontSize: 28, color: "#9fb0c6", fontWeight: 500 }}>
          github.com/qpiai/retreval
          <span style={{ margin: "0 16px", color: "#3a3060" }}>·</span>
          arXiv 2601.02880
        </div>
      </AbsoluteFill>
    </Stage>
  );
};

// ---------- composition root ------------------------------------------------

export const Demo: React.FC<{ appFrames: number }> = ({ appFrames }) => {
  const { durationInFrames } = useVideoConfig();
  return (
    <AbsoluteFill style={{ background: "#06060b" }}>
      <Audio
        src={staticFile(BGM)}
        volume={(f) =>
          interpolate(
            f,
            [0, 20, durationInFrames - 40, durationInFrames],
            [0, 0.32, 0.32, 0],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
          )
        }
      />
      <Sequence durationInFrames={INTRO_FRAMES}>
        <Intro />
      </Sequence>
      <Sequence from={INTRO_FRAMES} durationInFrames={appFrames}>
        <AppClip durationInFrames={appFrames} />
      </Sequence>
      <Sequence from={INTRO_FRAMES + appFrames} durationInFrames={OUTRO_FRAMES}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
