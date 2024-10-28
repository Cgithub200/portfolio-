using Microsoft.AspNetCore.Mvc;
using VertoTest.Context;
using VertoTest.Models;
using System.Diagnostics;
using VertoTest.Services;
using System.IdentityModel.Tokens.Jwt;
using Microsoft.IdentityModel.Tokens;
using System.Security.Claims;
using System.Text;
using Microsoft.Extensions.Options;
using Azure.Core;




namespace VertoTest.Controllers
{
    public class DayController : Controller
    {
        private readonly EventInterface eventInterface;
        private readonly JwtSettings jwtSettings;
        public DayController(EventInterface eventInterface, IOptions<JwtSettings> jwtSettings)
        {
            this.eventInterface = eventInterface;
            this.jwtSettings = jwtSettings.Value;  // Access the JWT settings (including SecretKey) here
        }

        // Returns day view
        [HttpGet]
        public IActionResult Day()
        {
            return View();
        }

        // handles new event creation
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> AddEvent([FromBody] CalendarEvent content)
        {
            if (content.EventStart >= content.EventEnd)
            {
                return BadRequest("Start date after end date");
            }
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }
            await eventInterface.Add(content);
            return Json(new { success = true, message = "Event added successfully." });
        }

        // Returns a list of all events on a day
        [HttpGet]
        public async Task<IActionResult> EventList(DateTime queryDate)
        {
            var eventItems = await eventInterface.List(queryDate);
            var key = Encoding.ASCII.GetBytes(jwtSettings.SecretKey);
            var tokenHandler = new JwtSecurityTokenHandler();
            var tokens = eventItems.Select(eventItem => new
            {
                eventItem.EventStart,
                eventItem.EventEnd,
                eventItem.Title,
                eventItem.Description,
                Token = eventInterface.GenerateToken(tokenHandler, key, eventItem.EventID) // Token is sent instead of id
            }).ToList();
          
            return Json(tokens);
        }


        // Handles edit requests for events
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> EditEvent([FromBody] EditEventRequest request)
        {
            if (request.EventData.EventStart >= request.EventData.EventEnd)
            {
                return BadRequest("Start date after end date");
            }
            if (!ModelState.IsValid) 
            {
                return BadRequest(ModelState);
            }

            var key = Encoding.ASCII.GetBytes(jwtSettings.SecretKey); // Gets the event assosiated with the token
            var (result, errorMessage) = await eventInterface.ValidateAndRetrieveEventByToken(request.Token, key);

            if (!string.IsNullOrEmpty(errorMessage))
            {
                return BadRequest(errorMessage); 
            }
            var updateResult = await eventInterface.Edit(result, request);
            if (!updateResult) 
            {
                return StatusCode(500, "Update event failed");
            }
           
            return Ok(new { success = true, message = "Event update successfull." });
        }

        // Handles delete event requests
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DeleteEvent([FromBody] DeleteEventRequest request)
        {
            var key = Encoding.ASCII.GetBytes(jwtSettings.SecretKey); 
            ClaimsPrincipal principal;
            // Gets the event assosiated with the token
            var (result, errorMessage) = await eventInterface.ValidateAndRetrieveEventByToken(request.Token, key);

            if (!string.IsNullOrEmpty(errorMessage))
            {
                return BadRequest(errorMessage);
            }

            var updateResult = await eventInterface.Delete(result);
            if (!updateResult)
            {
                return StatusCode(500, "Delete event failed");
            }


            return Ok(new { success = true, message = "Event delete successfull." });
        }


       

    }
}
