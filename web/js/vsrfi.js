import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "VSRFI.VideoUpload",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name !== "VSRFI") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const node = this;
            const videoWidget = this.widgets.find(w => w.name === "video");

            // Video metadata from server
            let videoInfo = { fps: 30, frame_count: 0, duration: 0 };

            // --- File input for upload ---
            const fileInput = document.createElement("input");
            Object.assign(fileInput, {
                type: "file",
                accept: "video/webm,video/mp4,video/x-matroska,video/quicktime,video/x-msvideo,image/gif",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        await doUpload(fileInput.files[0]);
                    }
                }
            });
            document.body.append(fileInput);

            async function doUpload(file) {
                const body = new FormData();
                body.append("image", file);
                try {
                    node.progress = 0;
                    const resp = await api.fetchApi("/upload/image", {
                        method: "POST",
                        body
                    });
                    node.progress = undefined;
                    if (resp.status === 200) {
                        const data = await resp.json();
                        const filename = data.name;
                        if (!videoWidget.options.values.includes(filename)) {
                            videoWidget.options.values.push(filename);
                        }
                        videoWidget.value = filename;
                        videoWidget.callback?.(filename);
                    }
                } catch (e) {
                    node.progress = undefined;
                    console.error("VSRFI upload failed:", e);
                }
            }

            // --- Upload button widget ---
            const uploadBtn = this.addWidget("button", "choose video to upload", null, () => {
                app.canvas.node_widget = null;
                fileInput.click();
            });
            uploadBtn.options.serialize = false;

            // --- Drag and drop ---
            this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.('Files');
            this.onDragDrop = async (e) => {
                if (!e?.dataTransfer?.types?.includes?.('Files')) return false;
                const file = e.dataTransfer?.files?.[0];
                if (file && (file.type.startsWith('video/') || file.type === 'image/gif')) {
                    await doUpload(file);
                    return true;
                }
                return false;
            };

            // --- Video preview widget ---
            const container = document.createElement("div");
            container.style.width = "100%";

            const previewWidget = this.addDOMWidget("videopreview", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return container.value; },
                setValue(v) { container.value = v; },
            });

            previewWidget.parentEl = document.createElement("div");
            previewWidget.parentEl.style.width = "100%";
            container.appendChild(previewWidget.parentEl);

            const videoEl = document.createElement("video");
            videoEl.controls = false;
            videoEl.loop = true;
            videoEl.muted = true;
            videoEl.autoplay = true;
            videoEl.style.width = "100%";
            videoEl.hidden = true;
            previewWidget.parentEl.appendChild(videoEl);

            previewWidget.computeSize = function (width) {
                if (this.aspectRatio && !videoEl.hidden) {
                    let height = (node.size[0] - 20) / this.aspectRatio + 10;
                    if (!(height > 0)) height = 0;
                    return [width, height + 10];
                }
                return [width, -4];
            };

            videoEl.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = videoEl.videoWidth / videoEl.videoHeight;
                fitHeight(node);
            });

            videoEl.addEventListener("error", () => {
                videoEl.hidden = true;
                previewWidget.parentEl.hidden = true;
                fitHeight(node);
            });

            // Forward pointer events to canvas so node remains interactive over the preview
            for (const evt of ['contextmenu', 'pointerdown', 'pointermove', 'pointerup']) {
                container.addEventListener(evt, (e) => {
                    e.preventDefault();
                    const handlers = {
                        'contextmenu': '_mousedown_callback',
                        'pointerdown': '_mousedown_callback',
                        'pointermove': '_mousemove_callback',
                        'pointerup': '_mouseup_callback'
                    };
                    return app.canvas[handlers[evt]]?.(e);
                }, true);
            }
            container.addEventListener('mousewheel', (e) => {
                e.preventDefault();
                return app.canvas._mousewheel_callback?.(e);
            }, true);

            // Allow drag onto preview area
            container.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = "copy";
                app.dragOverNode = node;
            });

            // --- Query video metadata from server ---
            async function queryVideoInfo(filename) {
                if (!filename) return;
                try {
                    const params = new URLSearchParams({ filename, type: "input" });
                    const resp = await api.fetchApi("/vsrfi/video_info?" + params);
                    if (resp.status === 200) {
                        videoInfo = await resp.json();
                    }
                } catch (e) {
                    console.error("VSRFI: failed to query video info:", e);
                }
            }

            // --- Constrain playback to skip/cap range ---
            function updatePreviewRange() {
                if (!videoInfo.fps || videoEl.hidden || !videoEl.src) return;

                const skipWidget = node.widgets.find(w => w.name === "skip_first_frames");
                const capWidget = node.widgets.find(w => w.name === "frame_load_cap");

                const skip = skipWidget?.value || 0;
                const cap = capWidget?.value || 0;

                const startTime = skip / videoInfo.fps;
                let endTime = videoInfo.duration;
                if (cap > 0) {
                    endTime = Math.min(endTime, startTime + cap / videoInfo.fps);
                }

                // Seek to start of selected range
                videoEl.currentTime = startTime;

                // Set up looping within the selected range
                videoEl.ontimeupdate = () => {
                    if (videoEl.currentTime >= endTime) {
                        videoEl.currentTime = startTime;
                    }
                };
            }

            // --- Update preview when video selection changes ---
            function updatePreview(filename) {
                if (!filename) {
                    videoEl.hidden = true;
                    previewWidget.parentEl.hidden = true;
                    fitHeight(node);
                    return;
                }
                const ext = filename.split('.').pop().toLowerCase();
                const format = "video/" + ext;
                const params = new URLSearchParams({
                    filename: filename,
                    type: "input",
                    format: format
                });
                videoEl.src = api.apiURL("/view?" + params);
                videoEl.hidden = false;
                previewWidget.parentEl.hidden = false;
                fitHeight(node);

                // Query metadata then apply range constraints
                queryVideoInfo(filename).then(() => updatePreviewRange());
            }

            // Hook into the video widget's callback
            const origCallback = videoWidget.callback;
            videoWidget.callback = function (value) {
                origCallback?.apply(this, arguments);
                updatePreview(value);
            };

            // Hook into skip_first_frames and frame_load_cap widgets
            for (const widgetName of ['skip_first_frames', 'frame_load_cap']) {
                const w = this.widgets.find(w => w.name === widgetName);
                if (w) {
                    const origCb = w.callback;
                    w.callback = function (value) {
                        origCb?.apply(this, arguments);
                        updatePreviewRange();
                    };
                }
            }

            // Show preview if a video is already selected on load
            if (videoWidget.value) {
                setTimeout(() => updatePreview(videoWidget.value), 100);
            }

            // Cleanup on node removal
            const origRemoved = this.onRemoved;
            this.onRemoved = function () {
                fileInput?.remove();
                origRemoved?.apply(this, arguments);
            };
        };
    }
});
