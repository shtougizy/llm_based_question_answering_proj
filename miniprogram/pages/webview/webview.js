Page({
  data: { src: '' },
  onLoad(options) {
    if (options.src) {
      this.setData({ src: decodeURIComponent(options.src) });
    }
  }
});
