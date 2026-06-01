system_prompt="""Our task is to rank six textured 3D outputs each created by transferring texture from a style mesh onto a structure mesh. 
We will evaluate the results based on the visual quality of the texture transfer, using a single rendered view of each mesh. All images are 
rendered from the same viewpoint. While only one image is available per mesh, please imagine each as a full 3D object — consider how textures might wrap around 
and behave across surfaces in three dimensions. The method should not only transfer color and texture from the style mesh, but also do so with 
semantic awareness — applying textures meaningfully to appropriate parts of the structure. Additionally, the geometry of the structure mesh 
must be preserved and visually clear.

# Instruction
We Would like to compare based on the following criteria:

1. Style Fidelity
How well does the output capture the visual essence of the style mesh? Look for color accuracy, material feel (e.g., matte, glossy, metallic), and stylistic patterns or motifs. A strong transfer should respect the semantic intent of the texture — e.g., wood textures mapped to handles, not faces. The result should feel like a coherent restyling, not a random recoloring.

2. Structure Clarity
Does the texture preserve the recognizable geometry of the structure mesh? Key parts (arms, legs, surfaces, joints) should remain distinguishable. Textures should enhance, not obscure. Imagine rotating the object: would the structure remain clear? Preservation of 3D form and part boundaries is critical.

3. Style Integration
How smoothly and appropriately is the style applied to the structure? Evaluate transitions across surfaces, texture seams, and alignment with part boundaries. Semantically aware integration maps the right textures to the right parts. Bad integration looks pasted or mismatched.

4. Detail Quality
Are local textures (grains, brushwork, ornamentation) clean, sharp, and artifact-free? Look for noise, blur, or visual inconsistencies. Even with stylization, the details should feel intentional and uniformly high quality across the mesh.

5. Shape Adaptation
Does the texture naturally follow the 3D geometry? Look for flow along curves, alignment to contours, and absence of warping or stretching. Imagine wrapping the style across a full 3D object. Well-adapted textures maintain realism and part continuity.

6. Overall Quality
Considering all of the above, which output delivers the best result overall? Look at visual appeal, technical execution, and whether the texture feels intentionally and coherently applied to the structured shape.

# Output Format
For each criterion, rank outputs from best (1) to worst (6). Ties are allowed (e.g., 2 2 3 4 5 6) but use sparingly.

Summarize in this format:
Final answer: rankA / rankB / rankC / rankD / rankE / rankF
(Style Fidelity / Structure Clarity / Style Integration / Detail Quality / Shape Adaptation / Overall)

Example:
4 1 2 3 5 6 / 2 3 1 4 5 6 / 1 2 2 3 5 6 / 2 1 3 4 5 6 / 1 3 4 2 5 6 / 1 2 3 4 5 6
"""