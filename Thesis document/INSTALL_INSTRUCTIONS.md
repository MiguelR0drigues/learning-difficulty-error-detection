# LaTeX Installation Instructions

## Step 1: Uninstall MacTeX

Run this command in your terminal (it will ask for your password):

```bash
brew uninstall --cask mactex
```

## Step 2: Install BasicTeX (minimal LaTeX distribution)

After MacTeX is uninstalled, run:

```bash
brew install --cask basictex
```

## Step 3: Set up PATH and install required packages

After BasicTeX is installed, restart your terminal or run:

```bash
eval "$(/usr/libexec/path_helper)"
export PATH="/Library/TeX/texbin:$PATH"
```

Then install the required LaTeX packages:

```bash
sudo tlmgr update --self
sudo tlmgr install latexmk biber biblatex biblatex-apa tikz pgfplots glossaries makecell
```

## Step 4: Build your document

Once everything is installed, you can build your PDF:

```bash
make
```

Or use the build script:

```bash
./build.sh
```

## Alternative: Keep MacTeX

If you prefer to keep MacTeX (it's already installed), just restart your terminal and the PATH should be set automatically. Then you can run `make` directly.

