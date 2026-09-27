# README #

*TMDEI Dissertation  LaTeX Template - Version 0.1 (Dec/2015)*

This template explains the main formatting rules to apply to a Master Dissertation work for TMDEI, of the MSc in Computer Engineering of the Computer Engineering Department (DEI) of the School of Engineering (ISEP) of the Polytechnic of Porto (IPP).

**You can fork this repository to make your own dissertation based on this template.**

This template is based on MastersDoctoralThesis version 1.2 by Vel (vel@latextemplates.com) and Johannes Böttcher, downloaded from [LaTeXTemplates](http://www.LaTeXTemplates.com) in November/2015. Adapted to TMDEI/ISEP style (Dec/2015) by Nuno Pereira and Paulo Baltarejo (DEI/ISEP).

## How do I get set up? ##

Just fork the repository and use it. You will need LaTeX tools installed in your system, with the packages needed by the template (more details bellow).

### LaTeX Distribution

LaTeX is available for many systems including Windows, Linux and Mac OS X. Check the webpage for the LaTex project for more information: <https://latex-project.org/ftp.html>.

Make sure you have the following tools installed: **pdflatex**, **makeglossaries**, **biber**, **latexmk**.

### LaTeX Packages Needed

| Package | Obs |
|---------|-----|
|babel|Required for automatically changing names of document elements to languages besides english|
|scrbase|Required for handling language-dependent names of sections/document elements|
|scrhack|Loads fixes for various packages|
|setspace|Required for changing line spacing|
|longtable|Required for tables that span multiple pages (used in the symbols, abbreviations and physical constants pages)|
|siunitx|Required for \SI commands|
|graphicx|Required to include images|
|xcolor|Required for extra color names|
|booktabs|Required for better table rules|
|inputenc|Required for inputting portuguese characters|
|fontenc|Output font encoding for portuguese characters|
|csquotes|Required to generate language-dependent quotes in the bibliography|
|cmbright|Default font: CM Bright, lighter sans-serif variant of Computer Modern Sans Serif|
|algorithm|Required for algorithms|
|algpseudocode|Part of algorithmicx package, required to customize the layout of algorithms|
|listings|Required for code listings|
|glossaries|Required to define acronyms and make glossaries|
|caption|Required for customising the captions|
|biblatex|Required for citations and bibliography|
|tikz|Required for creating graphics programmatically (can be removed if not used)|
|pgfplots|Required for drawing high--quality function plots (can be removed if not used)|

## Hot Reload / Live Preview ##

The project includes a hot reload feature that automatically rebuilds the PDF when you save changes to your `.tex` files.

### Quick Start

Run the following command in your terminal:

```bash
make watch
```

or

```bash
make preview
```

This will:
1. Build the PDF initially
2. Open it in your PDF viewer
3. Watch for changes to any `.tex` files
4. Automatically rebuild when you save changes
5. Refresh the PDF viewer (works best with Skim)

Press `Ctrl+C` to stop the watch mode.

### Recommended PDF Viewer: Skim

For the best experience with auto-refresh, install [Skim](https://skim-app.sourceforge.io/), a free PDF viewer for macOS designed for LaTeX users.

After installing Skim:
1. Open Skim
2. Go to **Skim → Preferences → Sync**
3. Check **"Check for file changes"** and **"Reload automatically"**

Then update `latexmk.rc` to use Skim by uncommenting the Skim line and commenting the Preview line.

### Alternative: Using Preview

The default configuration uses macOS Preview. While it will open the PDF, you may need to manually refresh (⌘R) when the PDF is updated. Skim provides a better experience with automatic refresh.

## Who do I talk to? ##

* Nuno Pereira (nap@isep.ipp.pt) and 
* Paulo Baltarejo (pbs@isep.ipp.pt)