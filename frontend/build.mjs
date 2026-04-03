import * as esbuild from 'esbuild';

const watch = process.argv.includes('--watch');

const buildOptions = {
  entryPoints: ['src/main.jsx'],
  bundle: true,
  format: 'esm',
  outfile: 'app.js',
  jsx: 'automatic',
  target: ['es2020'],
  loader: {
    '.js': 'jsx',
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify(watch ? 'development' : 'production'),
  },
  minify: !watch,
  sourcemap: watch,
};

if (watch) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log('Watching frontend source for changes...');
} else {
  await esbuild.build(buildOptions);
  console.log('Built frontend/app.js');
}
