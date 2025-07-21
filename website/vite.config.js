import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
// https://vitejs.dev/config/
export default defineConfig({
	plugins: [react()],
	build: {
		target: "es2022",
	},
	esbuild: {
		target: "es2022",
	},
	optimizeDeps: {
		esbuildOptions: {
			target: "es2022",
		},
	},
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "./src"),
		},
	},
	server: {
		port: 5173,
		host: true,
		strictPort: true
	},
	preview: {
		port: 3000,
		host: '0.0.0.0',
		strictPort: true
	}
});
