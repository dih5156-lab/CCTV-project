Put registered face images in this folder.

Example:
- known_faces/alice.jpg
- known_faces/bob.jpg

Then reference them from ../known_faces.json like this:
[
  { "name": "alice", "image": "known_faces/alice.jpg" },
  { "name": "bob", "image": "known_faces/bob.jpg" }
]
