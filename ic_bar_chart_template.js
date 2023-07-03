var {{chart_id}} = echarts.init(document.getElementById("{{chart_id}}"));

{{chart_id}}.setOption({
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