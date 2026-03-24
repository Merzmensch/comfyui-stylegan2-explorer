import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "StyleGAN2.AutoWalk",
    async nodeCreated(node) {
        if (node.comfyClass !== "StyleGAN2LatentWalk") return;

        // Add Walk / Stop button directly on the node
        const btn = node.addWidget("button", "▶ Start Walk", null, () => {
            if (btn.value === "▶ Start Walk") {
                btn.value = "■ Stop Walk";
                btn.name  = "■ Stop Walk";
                window._sg2_walking = true;
                window._sg2_fps     = 6;
                walkLoop();
            } else {
                btn.value = "▶ Start Walk";
                btn.name  = "▶ Start Walk";
                window._sg2_walking = false;
            }
            node.setDirtyCanvas(true);
        });

        // FPS slider
        node.addWidget("slider", "Walk FPS", 6, (v) => {
            window._sg2_fps = v;
        }, { min: 1, max: 20, step: 1 });

        function walkLoop() {
            if (!window._sg2_walking) return;
            app.queuePrompt(0, 1);   // queue one generation
            setTimeout(walkLoop, 1000 / (window._sg2_fps || 6));
        }
    }
});