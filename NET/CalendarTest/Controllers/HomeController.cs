using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using VertoTest.Models;
using VertoTest.Services;

namespace VertoTest.Controllers
{
    public class HomeController(ILogger<HomeController> logger, EventInterface eventInterface) : Controller
    {
        private readonly ILogger<HomeController> logger = logger;
        private readonly EventInterface _eventInterface = eventInterface;

        public IActionResult HomePage()
        {
            return View();
        }



        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }


        [HttpGet]
        public async Task<IActionResult> checkMonthEvents(DateTime firstDay, DateTime lastDay)
        {
            var response = await _eventInterface.CheckDaysWithEventsInRange(firstDay, lastDay);
            Debug.WriteLine(response);
            return Json(response);
        }
    }
}
