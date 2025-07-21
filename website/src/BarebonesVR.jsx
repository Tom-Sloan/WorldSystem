// This file contains a React component that renders a barebones VR experience
// Prompt: Add a new tab to show a barebones VR demo from an HTML example
import React, { useEffect, useRef } from 'react';

function BarebonesVR() {
  const containerRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current) return;

    // Define required scripts directly in the component
    const script = document.createElement('script');
    script.textContent = `
      (function () {
      'use strict';

      // XR globals.
      let xrButton = document.getElementById('xr-button');
      let xrSession = null;
      let xrRefSpace = null;

      // WebGL scene globals.
      let gl = null;

      // Checks to see if WebXR is available and, if so, requests an XRDevice
      // that is connected to the system and tests it to ensure it supports the
      // desired session options.
      function initXR() {
        // Is WebXR available on this UA?
        if (navigator.xr) {
          // If the device allows creation of exclusive sessions set it as the
          // target of the 'Enter XR' button.
          navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
            if (supported) {
              // Updates the button to start an XR session when clicked.
              xrButton.addEventListener('click', onButtonClicked);
              xrButton.textContent = 'Enter VR';
              xrButton.disabled = false;
            }
          });
        }
      }

      // Called when the user clicks the button to enter XR. If we don't have a
      // session we'll request one, and if we do have a session we'll end it.
      function onButtonClicked() {
        if (!xrSession) {
          navigator.xr.requestSession('immersive-vr').then(onSessionStarted);
        } else {
          xrSession.end();
        }
      }

      // Called when we've successfully acquired a XRSession. In response we
      // will set up the necessary session state and kick off the frame loop.
      function onSessionStarted(session) {
        xrSession = session;
        xrButton.textContent = 'Exit VR';

        // Listen for the sessions 'end' event so we can respond if the user
        // or UA ends the session for any reason.
        session.addEventListener('end', onSessionEnded);

        // Create a WebGL context to render with, initialized to be compatible
        // with the XRDisplay we're presenting to.
        let canvas = document.createElement('canvas');
        gl = canvas.getContext('webgl', { xrCompatible: true });

        // Use the new WebGL context to create a XRWebGLLayer and set it as the
        // sessions baseLayer. This allows any content rendered to the layer to
        // be displayed on the XRDevice.
        session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });

        // Get a reference space, which is required for querying poses. In this
        // case an 'local' reference space means that all poses will be relative
        // to the location where the XRDevice was first detected.
        session.requestReferenceSpace('local').then((refSpace) => {
          xrRefSpace = refSpace;

          // Inform the session that we're ready to begin drawing.
          session.requestAnimationFrame(onXRFrame);
        });
      }

      // Called either when the user has explicitly ended the session by calling
      // session.end() or when the UA has ended the session for any reason.
      // At this point the session object is no longer usable and should be
      // discarded.
      function onSessionEnded(event) {
        xrSession = null;
        xrButton.textContent = 'Enter VR';

        // In this simple case discard the WebGL context too, since we're not
        // rendering anything else to the screen with it.
        gl = null;
      }

      // Called every time the XRSession requests that a new frame be drawn.
      function onXRFrame(time, frame) {
        let session = frame.session;

        // Inform the session that we're ready for the next frame.
        session.requestAnimationFrame(onXRFrame);

        // Get the XRDevice pose relative to the reference space we created
        // earlier.
        let pose = frame.getViewerPose(xrRefSpace);

        // Getting the pose may fail if, for example, tracking is lost. So we
        // have to check to make sure that we got a valid pose before attempting
        // to render with it. If not in this case we'll just leave the
        // framebuffer cleared, so tracking loss means the scene will simply
        // disappear.
        if (pose) {
          let glLayer = session.renderState.baseLayer;

          // If we do have a valid pose, bind the WebGL layer's framebuffer,
          // which is where any content to be displayed on the XRDevice must be
          // rendered.
          gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);

          // Update the clear color so that we can observe the color in the
          // headset changing over time.
          gl.clearColor(Math.cos(time / 2000),
                        Math.cos(time / 4000),
                        Math.cos(time / 6000), 1.0);

          // Clear the framebuffer
          gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

          // Normally you'd loop through each of the views reported by the frame
          // and draw them into the corresponding viewport here, but we're
          // keeping this sample slim so we're not bothering to draw any
          // geometry.
          /*for (let view of pose.views) {
            let viewport = glLayer.getViewport(view);
            gl.viewport(viewport.x, viewport.y,
                        viewport.width, viewport.height);

            // Draw a scene using view.projectionMatrix as the projection matrix
            // and view.transform to position the virtual camera. If you need a
            // view matrix, use view.transform.inverse.matrix.
          }*/
        }
      }

      // Start the XR application.
      initXR();
    })();
    `;

    // Add some basic styling to mimic the original CSS
    const style = document.createElement('style');
    style.textContent = `
      .barebones-button {
        display: inline-block;
        margin: 0;
        padding: 5px 10px;
        border: 1px solid #fff;
        border-radius: 4px;
        background-color: #2563eb;
        color: #fff;
        font: 13px sans-serif;
        text-align: center;
        cursor: pointer;
      }
      .barebones-button:hover {
        background-color: #1d4ed8;
      }
      .barebones-button:disabled {
        background-color: #ccc;
        border-color: #999;
        color: #999;
        cursor: default;
      }
      summary {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
        cursor: pointer;
      }
      details {
        margin-bottom: 20px;
      }
      .back {
        color: #2563eb;
        text-decoration: none;
        margin-left: 10px;
      }
      header {
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
      }
    `;

    // Append to the container
    containerRef.current.appendChild(style);
    
    // Create and add HTML structure
    const content = document.createElement('div');
    content.innerHTML = `
      <header>
        <details open>
          <summary>Barebones VR</summary>
          <p>
            This sample demonstrates extremely simple use of an "immersive-vr"
            session with no library dependencies. It doesn't render anything
            exciting, just clears your headset's display to a slowly changing
            color to prove it's working.
            <a class="back" href="#">Back</a>
          </p>
          <button id="xr-button" class="barebones-button" disabled>XR not found</button>
        </details>
      </header>
      <main style='text-align: center;'>
        <p>Click 'Enter VR' to see content</p> 
      </main>
    `;
    
    containerRef.current.appendChild(content);
    containerRef.current.appendChild(script);

    return () => {
      // Cleanup
      if (containerRef.current) {
        containerRef.current.innerHTML = '';
      }
    };
  }, []);

  return (
    <div 
      ref={containerRef} 
      style={{ 
        height: '100%', 
        width: '100%', 
        padding: '20px', 
        boxSizing: 'border-box', 
        backgroundColor: '#f5f5f5' 
      }}
    />
  );
}

export default BarebonesVR; 