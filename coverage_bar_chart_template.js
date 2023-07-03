var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
    grid: {
        top: '1.5%'
    },
    xAxis: {
      type: 'category',
      data: {{labels}},
      axisTick: {
          alignWithLabel: true
        }
    },
    yAxis: {
        type: 'value'
    },
    series: [
        {
            type: 'bar',
            data: {{data}}
        }
    ],
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    }
});

window.addEventListener('resize',function(){
  {{chart_id}}.resize();
})