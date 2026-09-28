"""Run from the repository root: streamlit run dashboard/app.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from agents.base import Policy, PolicyConfig, load_policy
from dungeon.entities import DungeonConfig, ACTIONS
from memory.runner import Session, paired
from memory.visualization import map_figure, hidden_figure, trajectory_figure
from rl.trainer import seed_everything

st.set_page_config(page_title="Dungeon With Amnesia · Memory Lab",page_icon="◈",layout="wide")
st.markdown("""<style>
html,body,[class*="css"] {font-family:'DM Sans',sans-serif}
.block-container {max-width:1580px;padding-top:2rem}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-.04em}
[data-testid="stSidebar"] {border-right:1px solid #253342}
.hero {padding:30px 36px;border:1px solid #314239;border-radius:18px;background:radial-gradient(ellipse at 90% 10%,#214236 0%,transparent 50%),linear-gradient(120deg,#15202a,#111820);margin-bottom:24px;position:relative;overflow:hidden}
.eyebrow {font-size:11px;letter-spacing:.23em;color:#7ed8bf;font-weight:700;text-transform:uppercase}
.hero h1 {font-size:46px;margin:8px 0 2px;font-weight:600;color:#f1f3ec;line-height:1.15}
.hero p {color:#a8b8c7;font-size:16px;margin:10px 0 0}
.badge {display:inline-block;background:#1c332e;color:#8fe5ca;border:1px solid #38594d;padding:5px 10px;border-radius:6px;font-size:11px;margin:16px 6px 0 0;letter-spacing:.06em}
.panel-label {font-size:11px;letter-spacing:.16em;color:#bba583;text-transform:uppercase;margin:12px 0}
.observation {background:#101923;border:1px solid #2c3a4a;border-left:3px solid #bba583;border-radius:10px;padding:22px;white-space:pre-wrap;line-height:1.75;font-size:14px;min-height:320px;color:#d5dfe9}
.event {border-left:2px solid #567789;background:#141f2b;padding:9px 12px;margin:7px 0;border-radius:0 6px 6px 0;font-size:12px;color:#b9c8d8}
.event.cue,.event.collected {border-color:#65e1c0;color:#a3edd6}
.event.contradiction {border-color:#ed9778;color:#ffc2a6}
.subtle {color:#8295a8;font-size:12px;line-height:1.6}
.action {padding:10px 15px;border:1px solid #385c51;background:#152a25;color:#8aebcd;border-radius:8px;font-size:13px;margin-bottom:12px}
[data-testid="stMetric"] {background:#131e2b;border:1px solid #263444;padding:14px 18px;border-radius:12px}
[data-testid="stMetricLabel"] {color:#a2b3c4;font-size:12px}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif;font-size:28px}
.stButton>button {border-radius:8px}
footer {visibility:hidden}
</style>""",unsafe_allow_html=True)

seed_everything(42)
if "next_seed" in st.session_state:
    st.session_state["map_seed"] = st.session_state.pop("next_seed")

@st.cache_resource
def get_policy(path, stamp, agent):
    if path:
        return load_policy(path)[0]
    return Policy(PolicyConfig(agent)).eval()

with st.sidebar:
    st.markdown("### ◈ AMNESIA LAB")
    st.caption("LOCAL INTELLIGENCE / MEMORY RESEARCH")
    st.divider()
    family = st.selectbox("Environment",["Controlled showcase","Procedural exploration"])
    agent = st.selectbox("Policy architecture",["gru","no_memory","history","lstm"],format_func=lambda a:{"gru":"GRU · recurrent memory","no_memory":"MLP · no memory","history":"History · last 4 decisions","lstm":"LSTM · recurrent memory"}[a])
    folder = "showcase" if family == "Controlled showcase" else "comparison"
    checkpoint = ROOT/"results"/folder/agent/"checkpoint.pt"
    custom = st.text_input("Optional checkpoint path",placeholder="Local .pt checkpoint")
    if custom:
        checkpoint = Path(custom)
    trained = checkpoint.is_file()
    if trained:
        st.success("Trained checkpoint loaded")
    else:
        st.warning("Untrained policy preview. Train this variant to measure performance.")
    seed = int(st.number_input("Dungeon seed",min_value=0,value=200000,step=1,key="map_seed"))
    distance = st.slider("Detour length",2,30,15,disabled=folder != "showcase")
    size = st.select_slider("Grid size",[5,6,8],value=5,disabled=folder == "showcase")
    reliability = st.slider("Clue reliability",0.,1.,.75,.05)
    speed = st.slider("Decisions per second",1,8,3)
    debug = st.toggle("Reveal ground truth",False)
    st.caption("Ground truth and the notebook are never inputs to the policy.")
    reset = st.button("↻ Reset episode",use_container_width=True)
    if st.button("＋ New dungeon",use_container_width=True):
        st.session_state["next_seed"] = seed+1
        st.rerun()
    st.divider()
    st.markdown("<div class='subtle'>A local POMDP laboratory.<br>PyTorch PPO · no API keys.<br>All reported results come from recorded runs.</div>",unsafe_allow_html=True)

cfg = DungeonConfig(size=size,rooms={5:8,6:15,8:20}[size],keys=1 if size == 5 else 3,
                    mode="showcase" if folder == "showcase" else "procedural",dependency=distance,
                    clue_reliability=reliability,max_steps=160)
policy = get_policy(str(checkpoint) if trained else "",checkpoint.stat().st_mtime if trained else 0,agent)
signature = (str(checkpoint),trained,agent,seed,str(cfg))
if reset or st.session_state.get("signature") != signature:
    st.session_state.session = Session(policy,cfg,seed)
    st.session_state.signature = signature
    st.session_state.running = False
    st.session_state.pop("pair",None)

st.markdown("""<div class="hero"><div class="eyebrow">Experiment 01 / The persistence of information</div>
<h1>Dungeon With Amnesia</h1><p>An agent that must remember to escape.</p>
<span class="badge">PARTIALLY OBSERVABLE</span><span class="badge">CAUSAL MEMORY INTERVENTIONS</span><span class="badge">LOCAL & REPRODUCIBLE</span></div>""",unsafe_allow_html=True)

live, comparison, results, protocol = st.tabs(["Explore the dungeon","Memory intervention","Experiment results","Research notes"])

def draw(fig):
    st.pyplot(fig,width="stretch")
    plt.close(fig)

with live:
    a,b,c,d = st.columns([1,1,1,3])
    if a.button("▶ Run agent",use_container_width=True,type="primary"):
        st.session_state.running = True
        st.rerun()
    if b.button("Ⅱ Pause",use_container_width=True):
        st.session_state.running = False
        st.rerun()
    if c.button("→ Step",use_container_width=True):
        st.session_state.running = False
        st.session_state.session.step()
    d.caption("Each decision reads only the current local observation and its assigned memory.")

    @st.fragment(run_every=1/speed if st.session_state.get("running") else None)
    def render_live():
        s = st.session_state.session
        if st.session_state.get("running") and not s.env.done:
            s.step()
        cols = st.columns(4)
        cols[0].metric("DECISIONS",s.env.steps)
        cols[1].metric("ROOMS DISCOVERED",len(s.trace.graph))
        cols[2].metric("EPISODE REWARD",f"{s.env.total_reward:+.2f}")
        cols[3].metric("STATUS","Escaped" if s.env.success else "Ended" if s.env.done else "Exploring")
        left,center,right = st.columns([1.2,1,.8])
        with left:
            st.markdown('<div class="panel-label">01 / Spatial notebook</div>',unsafe_allow_html=True)
            draw(map_figure(s,debug))
            st.caption("Numbers are observer-assigned room labels. Colored stubs are observed locked doors; unknown endpoints remain hidden.")
        with center:
            st.markdown('<div class="panel-label">02 / What the agent can see</div>',unsafe_allow_html=True)
            action = s.timeline[-1]["action"].upper() if s.timeline else "AWAITING FIRST DECISION"
            st.markdown(f'<div class="action">↳ {html.escape(action)}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="observation">{html.escape(s.obs.text())}</div>',unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel-label">03 / Observed event trace</div>',unsafe_allow_html=True)
            st.caption("An external notebook, not a decoding of the neural state.")
            for event in reversed(s.trace.events[-9:]):
                st.markdown(f'<div class="event {event["kind"]}"><b>t={event["step"]:02d}</b> · {html.escape(event["text"])}</div>',unsafe_allow_html=True)
        with st.expander("Learned state · heatmap, norm and PCA",expanded=False):
            draw(hidden_figure(s.timeline))
            draw(trajectory_figure(s.timeline))
        with st.expander("Decision timeline / human controls",expanded=False):
            if s.timeline:
                st.dataframe(pd.DataFrame(s.timeline)[["step","action","reward"]],hide_index=True,use_container_width=True)
            human = st.selectbox("Human action",ACTIONS)
            if st.button("Execute human action"):
                s.step(ACTIONS.index(human))
                st.rerun()
        if s.env.done and st.session_state.get("running"):
            st.session_state.running = False
            st.rerun()
    render_live()

with comparison:
    st.markdown("### Change the memory. Keep the world.")
    st.caption("Fork this exact episode before the next observation is encoded. Both branches use the same weights and greedy actions. Only the treatment's retained state changes.")
    c1,c2,c3 = st.columns(3)
    kind = c1.selectbox("Intervention",["reset","partial","noise","restore"],format_func=lambda k:{"reset":"Delete all memory","partial":"Zero part of the state","noise":"Add Gaussian noise","restore":"Restore an earlier state"}[k])
    strength = c2.slider("Fraction zeroed / noise standard deviation",0.,2.,.5,.05)
    s = st.session_state.session
    restore_step = c3.number_input("Restore snapshot at decision",min_value=0,max_value=max(0,len(s.snapshots)-1),value=0)
    x,y,z = st.columns(3)
    if x.button("Fork & compare from here",type="primary",disabled=s.env.done):
        control,treatment = s.fork(),s.fork()
        kwargs = {"fraction":min(1.,strength),"noise":strength,"seed":seed}
        if kind == "restore":
            kwargs["earlier"] = s.snapshots[int(restore_step)]
        treatment.apply(kind,**kwargs)
        st.session_state.pair = (control.finish(),treatment.finish())
    if y.button("Apply to live agent",disabled=s.env.done):
        kwargs = {"fraction":min(1.,strength),"noise":strength,"seed":seed}
        if kind == "restore":
            kwargs["earlier"] = s.snapshots[int(restore_step)]
        s.apply(kind,**kwargs)
        st.success(f"{kind.upper()} applied before decision {s.env.steps+1}")
    evidence = ROOT/"results/ablation/showcase_episode.json"
    if z.button("Replay verified deletion",disabled=not evidence.exists() or agent != "gru" or folder != "showcase"):
        recorded = json.loads(evidence.read_text())
        replay_cfg = DungeonConfig(**recorded["control"]["config"])
        st.session_state.pair = paired(policy,replay_cfg,recorded["seed"],recorded["at_step"])
    if agent == "no_memory":
        st.info("The no-memory agent has no retained state. These interventions cannot change its policy input.")
    if "pair" in st.session_state:
        pair = st.session_state.pair
        edit = pair[1].interventions[-1] if pair[1].interventions else None
        st.caption(f"Paired dungeon seed: {pair[0].env.seed} · intervention: {edit['kind']} after decision {edit['step']}" if edit else f"Paired dungeon seed: {pair[0].env.seed}")
        st.dataframe(pd.DataFrame([{"Branch":label,"Outcome":"ESCAPED" if run.env.success else "FAILED",
                                    "Decisions":run.env.steps,"Reward":round(run.env.total_reward,2)}
                                   for label,run in zip(["Control · intact","Treatment · modified"],pair)]),
                     hide_index=True,use_container_width=True)
        for col,label,run in zip(st.columns(2),["CONTROL · intact memory","TREATMENT · intervened memory"],pair):
            with col:
                st.markdown(f"#### {label}")
                m1,m2,m3 = st.columns(3)
                m1.metric("Outcome","ESCAPED" if run.env.success else "FAILED")
                m2.metric("Steps",run.env.steps)
                m3.metric("Reward",f"{run.env.total_reward:.2f}")
                draw(map_figure(run))
                draw(hidden_figure(run.timeline))
                st.dataframe(pd.DataFrame(run.timeline)[["step","action","reward"]],hide_index=True,use_container_width=True,height=250)
        diverged = next((i+1 for i,(a,b) in enumerate(zip(pair[0].timeline,pair[1].timeline)) if a["action_id"] != b["action_id"]),None)
        st.info(f"First action divergence: decision {diverged}" if diverged else "No action divergence in this pair. A state change alone does not establish a behavioral effect.")
        st.download_button("Download paired trace",json.dumps({"control":pair[0].record(),"treatment":pair[1].record()},indent=2),"paired_trace.json","application/json")
    with st.expander("Edit the notebook (visual-only)"):
        st.caption("Removing a notebook entry does not delete a neural memory. The policy never reads this log.")
        if s.trace.events:
            event_index = st.selectbox("Entry",range(len(s.trace.events)),format_func=lambda i:s.trace.events[i]["text"])
            if st.button("Remove displayed entry"):
                s.trace.remove_event(event_index)
                st.rerun()

with results:
    st.markdown("### Evidence, not expectations")
    st.caption("The initial runs use one training seed. Confidence intervals cover map sampling, not training variability. Quick procedural runs are a pipeline baseline, not a convergence claim.")
    group = st.radio("Experiment family",["showcase","comparison"],horizontal=True)
    file = ROOT/"results"/group/"comparison.csv"
    if file.exists():
        st.dataframe(pd.read_csv(file),hide_index=True,use_container_width=True)
    else:
        st.info("No comparison results recorded for this family yet. Run experiments/compare_memory.py.")
    paths = [ROOT/"results"/group/"training_curves.png",ROOT/"results"/group/"generalization/generalization.png",ROOT/"results"/group/"dependency/dependency_distance.png",ROOT/"results/ablation/memory_reset.png",ROOT/"results/ablation/memory_noise.png"]
    for path in paths:
        if path.exists():
            st.image(str(path),width="stretch")

with protocol:
    st.markdown("""### What does this experiment establish?
The environment contains information the agent needs later but cannot observe again. Successful behavior therefore depends on preserving useful information through time.

**Procedural exploration** uses seeded trees embedded in a grid, 1–4 matching key/door pairs, unreliable room clues, and a one-time exit seal. Policies see only local room features, passages, objects, inventory and currently visible clues. They never see coordinates or the observer's graph.

**Controlled showcase** uses one-way turnstiles to force a detour and return. Navigation is deliberately constrained so that the final rune decision isolates retention of a vanished cue. It demonstrates memory for the rune, **not learned semantic recollection of a door location**. Its topology is fixed; held-out seeds vary the rune, room descriptions and clues. Procedural results test unseen topologies separately.

**Fair comparison.** The MLP sees the current observation only. History sees the last four observation/action inputs. GRU/LSTM carry state across the episode and reset at episode boundaries. All share the same local action masks and reward settings. Parameter counts differ; architecture is part of the treatment.

**PPO state flow.** Collection carries state one decision at a time. Optimization replays complete episodes from zero state on each epoch, with no timestep shuffling. Padding is masked; true terminal states do not bootstrap; time limits do.

**Interpretation.** A randomized hidden-state intervention can destroy useful information but also shift the model off its training distribution. Paired success degradation supports dependence on the state, not a claim that a particular neuron stores a red door. The symbolic trace is an external diagnostic record.
""")
