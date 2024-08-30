# disco

## What is this?

An AI-powered prop from Disco Elysium.

Point the flashlight at something, press the button, and hear the result of a skill check from Harry's inner voices.

Click the image below to see a video of it in action.

[![YouTube video of the prop in action](https://img.youtube.com/vi/0x3jQzq0AlM/0.jpg)](https://www.youtube.com/watch?v=0x3jQzq0AlM)

## Development

### Loading the environment

```shell
$ direnv allow
```

```shell
$ disco
```

```shell
$ direnv reload
```

### Maintaining Python dependencies

```shell
$ poetry add some-package
```

```shell
$ poetry update
```

```shell
$ poetry up --latest
```

### Testing and linting

```shell
$ pre-commit run --all
```

### Using the Nix build system

```shell
$ nix run .
```

```shell
$ nix run '.#containerImage' | docker load
```

```shell
$ nix flake check
```

```shell
$ nix flake update
```

## Updating the base template

```shell
$ cruft update --checkout template
```
