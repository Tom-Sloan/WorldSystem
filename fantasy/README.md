# fantasy_world_builder

Something similar to what I was thinking: https://x.com/kimmonismus/status/1847977766588678501?s=10.

1. **You have a `.ply` file** representing the 3D geometry of a room.
2. **You also have the original images** (textures) that were used to create or map onto this `.ply` file.
3. **You want to generate a new 3D texture map** (a new skin) for this same `.ply` model—so it keeps all the same spatial dimensions, but looks completely different stylistically (e.g. you might transform it into a cyberpunk theme).

Essentially, you’re creating a “skin generator” that re-textures a room’s 3D model while preserving the exact layout (so the virtual dimensions remain accurate).
