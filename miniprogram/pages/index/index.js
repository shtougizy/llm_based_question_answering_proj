const API = require('../../utils/api');
const app = getApp();

Page({
  data: {
    textInput: '',
    imagePath: '',
    loading: false,
    loadingText: '解题中...',
    result: null,
    showThinking: false,
    markDone: false,
    isGuest: false,
    showCrop: false,
    cropRect: null,
    cropStart: null,
    imageInfo: null,
    canvasW: 0,
    canvasH: 0,
    user:null,
    needViz: false,          // 程序题可视化开关
    tipText: '',             // 当前显示的小知识
    vizUrl: '',        // 可视化的 data URL
    showViz: false,
    _tipTimer: null,
    ttsLoading: false,
    ttsPlaying: false,
  },

  onShow() {
    const app = getApp();
    this.setData({
      isGuest: !app.globalData.token || (app.globalData.user && app.globalData.user.role === 'guest'),
      user: app.globalData.user || { username: '未登录' }
    });
    this._loadTips();
  },
  
    // _loadTips 改为轮播
  _loadTips() {
    const tips = [
      "勾股定理：直角三角形两直角边的平方和等于斜边的平方。",
      "光速约为每秒30万千米，任何有质量的物体都无法达到光速。",
      "细胞是生命活动的基本单位，分为原核细胞和真核细胞两大类。",
      "化学元素周期表由门捷列夫于1869年提出，目前已发现118种元素。",
      "牛顿第一定律：物体在不受外力时保持静止或匀速直线运动状态。",
      "光合作用将光能转化为化学能，释放氧气，是地球生命的能量来源。",
      "DNA双螺旋结构由沃森和克里克于1953年发现，携带遗传信息。",
      "欧姆定律：电流等于电压除以电阻，I=U/R。",
      "质量守恒定律：化学反应前后各物质质量总和不变。",
      "相对论由爱因斯坦提出，揭示了质能等价关系 E=mc²。",
      "浮力定律：物体所受浮力等于排开液体的重力，由阿基米德发现。",
      "二进制是计算机使用的数制，只有0和1两个数字。",
      "递归算法通过函数调用自身解决问题，必须设置终止条件。",
      "动态规划将复杂问题分解为子问题，存储结果避免重复计算。",
      "时间复杂度O(n log n)是大多数排序算法能达到的最优平均复杂度。",
    ];
    // 随机起始位置
    let idx = Math.floor(Math.random() * tips.length);
    this.setData({ tipText: tips[idx] });

    // 每4秒随机换一条（避免顺序规律）
    const timer = setInterval(() => {
      let next;
      do { next = Math.floor(Math.random() * tips.length); } while (next === idx);
      idx = next;
      this.setData({ tipText: tips[idx] });
    }, 4000);
    this._tipTimer = timer;
  },

  // onUnload 时清除定时器
  onUnload() {
    if (this._tipTimer) clearInterval(this._tipTimer);
  },

  // _loadTips() {
  //   const tips = [
  //     "勾股定理：直角三角形两直角边的平方和等于斜边的平方。",
  //     "光速约为每秒30万千米，任何有质量的物体都无法达到光速。",
  //     "细胞是生命活动的基本单位，分为原核细胞和真核细胞两大类。",
  //     "化学元素周期表由门捷列夫于1869年提出，目前已发现118种元素。",
  //     "牛顿第一定律：一切物体在不受外力时保持静止或匀速直线运动状态。",
  //     "光合作用将光能转化为化学能，释放氧气，是地球生命的能量来源。",
  //     "DNA双螺旋结构由沃森和克里克于1953年发现，携带遗传信息。",
  //     "欧姆定律：电流等于电压除以电阻，I=U/R。",
  //     "质量守恒定律：化学反应前后各物质质量总和不变。",
  //     "相对论由爱因斯坦提出，揭示了质能等价关系E=mc²。",
  //     "浮力定律：物体所受浮力等于它排开液体的重力，由阿基米德发现。",
  //     "二进制是计算机使用的数制，只有0和1两个数字。",
  //     "递归算法通过函数调用自身来解决问题，需要设置终止条件。",
  //     "动态规划将复杂问题分解为子问题，通过存储子问题结果避免重复计算。",
  //     "时间复杂度O(n log n)是大多数排序算法能达到的最优平均复杂度。",
  //   ];
  //   this.setData({ tips, tipText: tips[Math.floor(Math.random() * tips.length)] });
  // },
  
  onVizChange(e) {
    this.setData({ needViz: e.detail.value });
  },

  // onShow() {
  //   const app = getApp();
  //   this.setData({
  //     isGuest: !app.globalData.token || (app.globalData.user && app.globalData.user.role === 'guest'),
  //     user: app.globalData.user || { username: '未登录' }
  //   });
  // },

  // onShow() {
  //   const user = app.globalData.user;
  //   this.setData({ isGuest: user && user.role === 'guest' });
  //   if (!app.globalData.token) {
  //     wx.navigateTo({ url: '/pages/login/login' });
  //   }
  // },

  // 拍照/选图
  // 选完图片后不直接搜题，先进入框选
  onChooseImage() {
    const self = this;
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      success(res) {
        const filePath = res.tempFiles[0].tempFilePath;
        // 直接进入裁剪，不弹对话框
        self._doCrop(filePath);
      }
    });
  },
  
  _doCrop(filePath) {
    const self = this;
    // 优先用 editImage（微信推荐，支持裁剪/旋转）
    if (wx.editImage) {
      wx.editImage({
        src: filePath,
        success(res) {
          self.setData({ imagePath: res.tempFilePath, result: null });
        },
        fail() {
          // editImage 失败则直接用原图
          self.setData({ imagePath: filePath, result: null });
        }
      });
    } else if (wx.cropImage) {
      wx.cropImage({
        src: filePath,
        cropScale: '0',
        success(res) {
          self.setData({ imagePath: res.tempFilePath, result: null });
        },
        fail() {
          self.setData({ imagePath: filePath, result: null });
        }
      });
    } else {
      // 不支持裁剪则直接用原图
      self.setData({ imagePath: filePath, result: null });
      wx.showToast({ title: '当前版本不支持裁剪', icon: 'none' });
    }
  },
  // onChooseImage() {
  //   const self = this;
  //   wx.chooseMedia({
  //     count: 1,
  //     mediaType: ['image'],
  //     sourceType: ['album', 'camera'],
  //     success(res) {
  //       const filePath = res.tempFiles[0].tempFilePath;
  //       // 先展示原图，让用户选择是否裁剪
  //       self.setData({ imagePath: filePath, result: null });
  //       wx.showModal({
  //         title: '选择操作',
  //         content: '是否框选题目区域？可提升识别准确率',
  //         confirmText: '框选题目',
  //         cancelText: '全图搜题',
  //         success(modal) {
  //           if (modal.confirm) {
  //             // 调用系统裁剪
  //             wx.cropImage({
  //               src: filePath,
  //               cropScale: '0',   // 0 表示自由比例
  //               success(cropRes) {
  //                 self.setData({ imagePath: cropRes.tempFilePath });
  //                 wx.showToast({ title: '已框选题目区域', icon: 'success' });
  //               },
  //               fail() {
  //                 wx.showToast({ title: '裁剪取消，使用全图', icon: 'none' });
  //               }
  //             });
  //           }
  //         }
  //       });
  //     }
  //   });
  // },

  // 清除图片
  onClearImage() {
    this.setData({ imagePath: '' });
  },

  // 输入题目文字
  onTextInput(e) {
    this.setData({ textInput: e.detail.value });
  },

  onImageLoad(e) {
    const { width, height } = e.detail;
    // 计算 canvas 尺寸（适配屏幕宽度）
    const screenW = wx.getSystemInfoSync().windowWidth;
    const scale = screenW / width;
    this.setData({
      canvasW: screenW,
      canvasH: Math.round(height * scale),
      imageInfo: { width, height, scale }
    });
  },
  
  // 开始拖拽
  onCropStart(e) {
    const { x, y } = e.touches[0];
    this.setData({ cropStart: { x, y }, cropRect: null });
  },
  
  // 拖拽中
  onCropMove(e) {
    const { cropStart, canvasW, canvasH } = this.data;
    if (!cropStart) return;
    const { x, y } = e.touches[0];
    const rect = {
      x: Math.max(0, Math.min(cropStart.x, x)),
      y: Math.max(0, Math.min(cropStart.y, y)),
      w: Math.abs(x - cropStart.x),
      h: Math.abs(y - cropStart.y),
    };
    this.setData({ cropRect: rect });
    this._drawCanvas(rect);
  },
  
  // 结束拖拽
  onCropEnd(e) {
    this.setData({ cropStart: null });
  },
  
  // 在 canvas 上绘制选框
  _drawCanvas(rect) {
    const ctx = wx.createCanvasContext('crop-canvas', this);
    const { canvasW, canvasH, imagePath } = this.data;
    ctx.drawImage(imagePath, 0, 0, canvasW, canvasH);
    // 半透明遮罩
    ctx.setFillStyle('rgba(0,0,0,0.4)');
    ctx.fillRect(0, 0, canvasW, canvasH);
    // 清除选框内遮罩
    ctx.clearRect(rect.x, rect.y, rect.w, rect.h);
    // 选框边框
    ctx.setStrokeStyle('#667eea');
    ctx.setLineWidth(3);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.draw();
  },
  
  // 确认框选，裁剪图片
  onCropConfirm() {
    const { cropRect, imagePath, imageInfo, canvasW } = this.data;
    if (!cropRect || cropRect.w < 20 || cropRect.h < 20) {
      // 没有框选，直接用整张图
      this.setData({ showCrop: false });
      return;
    }
    // canvas 坐标转原图坐标
    const scale = imageInfo.scale;
    const sx = Math.round(cropRect.x / scale);
    const sy = Math.round(cropRect.y / scale);
    const sw = Math.round(cropRect.w / scale);
    const sh = Math.round(cropRect.h / scale);
  
    wx.canvasToTempFilePath({
      canvasId: 'crop-canvas',
      x: cropRect.x, y: cropRect.y,
      width: cropRect.w, height: cropRect.h,
      destWidth: sw, destHeight: sh,
      success: (res) => {
        this.setData({
          imagePath: res.tempFilePath,  // 用裁剪后的图片
          showCrop: false,
          cropRect: null,
        });
      },
      fail: () => {
        wx.showToast({ title: '裁剪失败', icon: 'none' });
      }
    }, this);
  },
  
  // 跳过框选
  onCropSkip() {
    this.setData({ showCrop: false, cropRect: null });
  },


  // 提交搜题
  async onSubmit() {
    const { textInput, imagePath } = this.data;
    if (!textInput && !imagePath) {
      wx.showToast({ title: '请拍照或输入题目', icon: 'none' });
      return;
    }
  
    this.setData({ loading: true, loadingText: '上传中...', result: null });
  
    try {
      let data;
      if (imagePath) {
        const username = getApp().globalData.user?.username || 'default';
        const token = getApp().globalData.token;
        const baseUrl = getApp().globalData.baseUrl;
  
        // Step1: 上传图片，获取 task_id
        const uploadRes = await new Promise((resolve, reject) => {
          wx.uploadFile({
            url: baseUrl + '/api/search/image/async',
            filePath: imagePath,
            name: 'file',
            // formData: { username, need_visualization: 'false' },
            formData: { username, need_visualization: this.data.needViz ? 'true' : 'false' },
            header: { 'Authorization': token ? 'Bearer ' + token : '' },
            timeout: 30000,  // 上传本身只需30秒
            success: (res) => {
              try { resolve(JSON.parse(res.data)); }
              catch(e) { reject(new Error('上传返回格式错误')); }
            },
            fail: reject
          });
        });
  
        const taskId = uploadRes.task_id;
        this.setData({ loadingText: '识别中...' });
  
        // Step2: 轮询结果，最多等120秒
        data = await this._pollTask(taskId, 120);
  
      } else {
        // 文字搜题
        const API = require('../../utils/api');
        // data = await API.searchText(textInput);
        data = await API.searchText(textInput, this.data.needViz);
      }
  
      // 拿到结果后
      this.setData({ result: data, markDone: false });

      // 处理程序题可视化
      if (data.visualization_html) {
        const baseUrl = getApp().globalData.baseUrl;
        const token = getApp().globalData.token;
        // 把HTML传给后端，换取一个HTTPS访问URL
        wx.request({
          url: baseUrl + '/api/viz/store',
          method: 'POST',
          header: {
            'Content-Type': 'application/json',
            'Authorization': token ? 'Bearer ' + token : ''
          },
          data: { html: data.visualization_html },
          success: (res) => {
            if (res.data && res.data.key) {
              const vizUrl = baseUrl + '/api/viz/' + res.data.key;
              this.setData({ vizUrl, showViz: true });
            }
          },
          fail: () => {
            this.setData({ vizUrl: '', showViz: false });
          }
        });
      } else {
        this.setData({ vizUrl: '', showViz: false });
      }

      // // 处理程序题可视化：写临时文件，展示跳转按钮
      // if (data.visualization_html) {
      //   const fs = wx.getFileSystemManager();
      //   const tmpPath = wx.env.USER_DATA_PATH + '/viz_result.html';
      //   try {
      //     fs.writeFileSync(tmpPath, data.visualization_html, 'utf8');
      //     this.setData({ vizUrl: tmpPath, showViz: true });
      //   } catch(err) {
      //     this.setData({ vizUrl: '', showViz: false });
      //   }
      // } else {
      //   this.setData({ vizUrl: '', showViz: false });
      // }
  
    } catch(e) {
      wx.showToast({ title: e.message || '搜题失败', icon: 'none' });
    } finally {
      this.setData({ loading: false });
    }
  },
  
  // 轮询任务结果
  _pollTask(taskId, maxSeconds) {
    const baseUrl = getApp().globalData.baseUrl;
    const token = getApp().globalData.token;
    const self = this;
    let elapsed = 0;
  
    return new Promise((resolve, reject) => {
      const poll = () => {
        wx.request({
          url: `${baseUrl}/api/task/${taskId}`,
          header: { 'Authorization': token ? 'Bearer ' + token : '' },
          success: (res) => {
            const d = res.data;
            if (d.status === 'done') {
              resolve(d.result);
            } else if (d.status === 'error') {
              reject(new Error(d.error || '解题失败'));
            } else {
              // pending，继续等
              elapsed += 3;
              if (elapsed >= maxSeconds) {
                reject(new Error('解题超时，请重试'));
                return;
              }
              // 更新提示文字
              if (elapsed < 20) self.setData({ loadingText: '识别题目中...' });
              else if (elapsed < 60) self.setData({ loadingText: '解题中...' });
              else self.setData({ loadingText: `解题中...${elapsed}s` });
              setTimeout(poll, 3000);  // 每3秒轮询一次
            }
          },
          fail: () => {
            elapsed += 3;
            if (elapsed >= maxSeconds) reject(new Error('网络超时'));
            else setTimeout(poll, 3000);
          }
        });
      };
      setTimeout(poll, 3000);  // 3秒后开始轮询
    });
  },
  // async onSubmit() {
  //   const { textInput, imagePath } = this.data;
  //   if (!textInput && !imagePath) {
  //     wx.showToast({ title: '请拍照或输入题目', icon: 'none' });
  //     return;
  //   }

  //   this.setData({ loading: true, loadingText: imagePath ? '识别中...' : '解题中...', result: null });

  //   try {
  //     let data;
  //     if (imagePath) {
  //       // 图片搜题
  //       data = await new Promise((resolve, reject) => {
  //         const username = app.globalData.user ? app.globalData.user.username : 'default';
  //         const token = app.globalData.token;
  //         wx.uploadFile({
  //           url: app.globalData.baseUrl + '/api/search/image',
  //           filePath: imagePath,
  //           name: 'file',
  //           formData: { username, need_visualization: 'false' },
  //           header: { 'Authorization': token ? 'Bearer ' + token : '' },
  //           success: (res) => {
  //             try {
  //               resolve(JSON.parse(res.data));
  //             } catch(e) {
  //               reject(new Error('返回数据格式错误'));
  //             }
  //           },
  //           fail: reject
  //         });
  //       });
  //     } else {
  //       // 文字搜题
  //       data = await API.searchText(textInput);
  //     }

  //     this.setData({ result: data, markDone: false });

  //     // 游客首次搜题提示登录
  //     if (this.data.isGuest) {
  //       setTimeout(() => {
  //         wx.showModal({
  //           title: '登录后保存记录',
  //           content: '游客数据临时保存，登录后可永久保留学习记录',
  //           confirmText: '去登录',
  //           cancelText: '继续',
  //           success: (res) => {
  //             if (res.confirm) wx.navigateTo({ url: '/pages/login/login' });
  //           }
  //         });
  //       }, 1500);
  //     }
  //   } catch(e) {
  //     wx.showToast({ title: '搜题失败，请重试', icon: 'none' });
  //     console.error(e);
  //   } finally {
  //     this.setData({ loading: false });
  //   }
  // },

  // 折叠/展开思考过程
  toggleThinking() {
    this.setData({ showThinking: !this.data.showThinking });
  },

  // 标记错题
  async onMarkWrong() {
    const { result } = this.data;
    if (!result || !result.record_id) return;
    try {
      await API.markWrong(result.record_id);
      this.setData({ markDone: true });
      wx.showToast({ title: '已加入错题本', icon: 'success' });
    } catch(e) {
      wx.showToast({ title: '操作失败', icon: 'none' });
    }
  },

  // 打开可视化页面
  onOpenViz() {
    if (this.data.vizUrl) {
      wx.navigateTo({
        url: '/pages/webview/webview?src=' + encodeURIComponent(this.data.vizUrl)
      });
    }
  },

  onPlayTTS() {
    const self = this;
    const answer = this.data.result && this.data.result.llm_answer;
    if (!answer) return;
    if (this.data.ttsPlaying && this._audioCtx) {
      this._audioCtx.stop();
      this.setData({ ttsPlaying: false });
      return;
    }
    this.setData({ ttsLoading: true });
    const app = getApp();
    wx.request({
      url: app.globalData.baseUrl + '/api/tts',
      method: 'POST',
      header: { 'Content-Type': 'application/json',
        'Authorization': app.globalData.token ? 'Bearer ' + app.globalData.token : '' },
      data: { text: answer.slice(0, 400), voice: 'Junhao' },
      timeout: 120000,
      success(res) {
        if (!res.data || !res.data.audio_base64) {
          wx.showToast({ title: '无音频数据', icon: 'none' }); return;
        }
        const fs = wx.getFileSystemManager();
        const tmpPath = wx.env.USER_DATA_PATH + '/tts_out.wav';
        fs.writeFile({
          filePath: tmpPath, data: res.data.audio_base64, encoding: 'base64',
          success() {
            const ctx = wx.createInnerAudioContext();
            ctx.src = tmpPath; ctx.play();
            self._audioCtx = ctx;
            self.setData({ ttsPlaying: true });
            ctx.onEnded(() => self.setData({ ttsPlaying: false }));
            ctx.onError(() => { self.setData({ ttsPlaying: false });
              wx.showToast({ title: '播放失败', icon: 'none' }); });
          },
          fail() { wx.showToast({ title: '文件写入失败', icon: 'none' }); }
        });
      },
      fail() { wx.showToast({ title: '语音合成失败', icon: 'none', duration: 2000 }); },
      complete() { self.setData({ ttsLoading: false }); }
    });
  },

  // 重置
  onReset() {
    this.setData({ textInput: '', imagePath: '', result: null, markDone: false });
  },

  // 跳转用户中心/登录
  onUserTap() {
    if (app.globalData.token && !this.data.isGuest) {
      wx.showActionSheet({
        itemList: ['退出登录'],
        success: (res) => {
          if (res.tapIndex === 0) {
            app.globalData.token = '';
            app.globalData.user  = null;
            wx.removeStorageSync('token');
            wx.removeStorageSync('user');
            wx.navigateTo({ url: '/pages/login/login' });
          }
        }
      });
    } else {
      wx.navigateTo({ url: '/pages/login/login' });
    }
  }
});
