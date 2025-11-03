# Contributing Guide
A guide on how to setup, code, test, review, and release so contributions meet our Definition of Done.
## Code of Conduct
### Our Pledge
In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.
### Our standards
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Focus on what's best for the community
- Gracefully accept constructive criticism
## Getting Started
**Dependencies:**
- GNU12.2.0+ Tool Collection
- CMake 3.30.5+
- Make
- Linux environment
- Module tool manager

**Building MDStress as a linkable library:**
```sh
git clone https://github.com/vanegasj/mdstress-library.git  
cd mdtress-library  
mkdir build  
cd build  
module load cmake/3.30.5 gnu12/12.2.0  
cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/apps/mdstress-library/1.0  
make -j 8  
make install
```
Assuming that there were no issues during the compilation and installation, you should now have the library installed inside `$HOME/apps/mdstress-library/1.0` . 
Use the following commands to make the library available through the modules environment:
```
cd $HOME/apps/modulefiles  
mkdir mdstress-library  
cd mdtress-library
```
Use the following command to create a text file called 1.0 that has the following content
```
echo -e "#%Module1.0  
module load gnu12/12.2.0  
set root_dir $HOME/apps/mdstress-library/1.0  
prepend-path LD_LIBRARY_PATH \$root_dir/lib  
prepend-path MANPATH \$root_dir/share/man  
prepend-path PATH \$root_dir/bin  
prepend-path CMAKE_PREFIX_PATH \$root_dir" > 1.0
```
You can now use the locally installed library with the command
```
module load mdstress-library/1.0
```
Now that MDStress is built as a library, you can build GROMACS-LS linking your dev build of MDStress.
```sh
cd $HOME/apps/sources  
git clone https://github.com/vanegasj/gromacs-ls.git  
cd gromacs-ls  
mkdir build  
cd build  
module load cmake/3.30.5 gnu12/12.2.0 mdstress-library/1.0  
cmake ../ -DCMAKE_INSTALL_PREFIX=$HOME/apps/gromacs-ls/2016.3 -DGMX_BUILD_OWN_FFTW=ON  
make -j 8  
make install
```
Assuming that there were no issues during the compilation and installation, you should now have the library installed inside $HOME/apps/mdstress-library/1.0 . 
Use the following commands to make the library available through the modules environment:
```
cd $HOME/apps/modulefiles  
mkdir gromacs-ls  
cd gromacs-ls
```
Use the following command to create a text file called 1.0 that has the following content
```
echo -e "#%Module1.0  
module load gnu12/12.2.0 mdstress-library/1.0  
set root_dir $HOME/apps/gromacs-ls/2016.3  
prepend-path LD_LIBRARY_PATH \$root_dir/lib  
prepend-path MANPATH \$root_dir/share/man  
prepend-path PATH \$root_dir/bin  
prepend-path CMAKE_PREFIX_PATH \$root_dir"  
append-path GMXLIB $root_dir/share/gromacs/top:/apps/gmxff  
prepend-path DSSP $root_dir/bin/dssp > 2016.3
```
You can now use the locally installed library with the command
```
module load gromacs-ls/2016.3
```
The GROMACS-LS executable is called `gmx_LS`.
## Branching & Workflow
- We use a trunk based git workflow
- The default branch is capstone
- Any new feature should take place on a separate branch
- Rebase if using local, private branches to rewrite commits for clean history
- Merge into capstone when feature meets DoD
- Branches should follow this naming specification:
	- feature: `feature/some-feature`
	- bug: `bug/some-bug`
	- migration: `migration/some-migration`
## Issues & Planning
- Issues must be filed in the Linear Workspace
- If filing for a feature:
	- Issue must be labeled as a feature
	- Issue must be grouped with other issues within same feature group
- Issues must state requirements needed
- Issue must state time estimation
## Commit Messages
Use **Conventional Commits** (e.g., `feat: add login page`, `fix: correct null pointer`, `docs: update README`).  
Reference issues with `Closes #123` or `Fixes #456` at the end of the commit message.
## Code Style, Linting & Formatting
- No formatter/linter implemented.
- When pushing new files:
	- Refer to https://google.github.io/styleguide/cppguide.html
- When modifying files:
	- Try to keep new lines/functional blocks to this style guide: https://google.github.io/styleguide/cppguide.html
	- Do not spend dev time re-formatting current files
## Testing
- Unit tests required when modifiying any function, and when creating any new function
- Building MDStress generates executable test files using hard-stop assert statements in the `./build/test/` directory
- Building GROMACS-LS runs built-in testing
- In order to minimize dependencies for legacy users, there is no testing framework implemented
## Pull Requests & Reviews
- Use the PR template (`.github/pull_request_template.md`)
- Provide clear description of changes
- Provide clear and concise summary
- Keep document focused
- At least one reviewer approval and all status checks (CI/CD), if applicable, must pass
## CI/CD
- No CI/CD implemented as of yet
## Security & Secrets
- If a security vulnerability is found, please report directly to Juan Vanegas (vanegasj@oregonstate.edu)
- There are no secret variables in this project, but if there were do not hardcode
## Documentation Expectations
- Document in concise and clear comments throughout code
- Prefer clear code over verbose comments
- Update any .md file if it becomes out-of-date
## Release Process
- Release Process not specified (blocked since we may be porting GROMACS-LS to a new version of GROMACS)
## Support & Contact
- For general questions/issues, create an issue in github
- Contact Juan Vanegas (vanegasj@oregonstate.edu) if any additional support is needed. Expect a response window of about a week

