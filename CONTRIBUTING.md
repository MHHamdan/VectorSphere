# Contributing to VectorHub

Thank you for your interest in contributing to VectorHub! This document provides guidelines and steps for contributing to this project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/MHHamdan/VectorHub.git
   cd VectorHub
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Update documentation if needed
5. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```

### Pull Request Process

1. Push to your fork
2. Submit a Pull Request (PR)
3. Ensure PR description clearly describes the problem and solution
4. Include relevant issue numbers if applicable

### Code Standards

- Follow PEP 8 style guide
- Write meaningful commit messages
- Include docstrings for functions and classes
- Add tests for new features
- Keep functions focused and modular
- Use type hints when possible

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Include both positive and negative test cases

## Development Workflow

1. Choose an issue to work on (or create one)
2. Comment on the issue to let others know you're working on it
3. Create a branch and implement your changes
4. Submit a PR
5. Address any review comments
6. Once approved, your PR will be merged

## Questions or Need Help?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Suggestions for improvement

## License
By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.
